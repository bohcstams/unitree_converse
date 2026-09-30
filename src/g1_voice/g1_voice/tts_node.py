import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import threading
import os
import io
import wave
import subprocess
import tempfile
import numpy as np
import sounddevice as sd


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        self.declare_parameter('voice_model_path',
            os.path.expanduser('~/.local/share/piper/en_US-lessac-medium.onnx'))
        self.declare_parameter('use_cuda', False)
        self.declare_parameter('output_device', -1)
        self.declare_parameter('continuous_mode', False)
        self.declare_parameter('output_sample_rate', 44100)
        self.declare_parameter('tts_mode', 'python')  # 'python' or 'binary'
        self.declare_parameter('piper_binary', '/usr/local/bin/piper')

        self.model_path = self.get_parameter('voice_model_path').get_parameter_value().string_value
        self.use_cuda = self.get_parameter('use_cuda').get_parameter_value().bool_value
        self.output_device = self.get_parameter('output_device').get_parameter_value().integer_value
        self.output_sample_rate = self.get_parameter('output_sample_rate').get_parameter_value().integer_value
        self.tts_mode = self.get_parameter('tts_mode').get_parameter_value().string_value
        self.piper_binary = self.get_parameter('piper_binary').get_parameter_value().string_value

        if self.output_device == -1:
            self.output_device = self._resolve_default_output()
        self.continuous_mode = self.get_parameter('continuous_mode').get_parameter_value().bool_value

        self.voice = None
        self.is_speaking = False
        self.speak_lock = threading.Lock()

        if self.tts_mode == 'python':
            self._load_voice_python()
        else:
            self._check_binary()

        self.response_sub = self.create_subscription(
            String, 'llm_response', self.response_callback, 10)
        self.create_subscription(Bool, '/g1/voice/continuous_start', self._continuous_start_cb, 10)
        self.create_subscription(Bool, '/g1/voice/continuous_stop', self._continuous_stop_cb, 10)

        self.status_pub = self.create_publisher(String, '/g1/tts/status', 10)
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)

        self.get_logger().info(f'TTS node ready — mode: {self.tts_mode}, model: {self.model_path}')

    def _load_voice_python(self):
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Piper model not found: {self.model_path}')
            return
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
            self.get_logger().info(
                f'Piper Python voice loaded — native rate: {self.voice.config.sample_rate}Hz')
        except Exception as e:
            self.get_logger().error(f'Failed to load Piper Python voice: {e}')

    def _check_binary(self):
        if not os.path.exists(self.piper_binary):
            self.get_logger().error(f'Piper binary not found: {self.piper_binary}')
        else:
            self.get_logger().info(f'Piper binary found: {self.piper_binary}')

    def response_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            reply = data.get('reply', '').strip() if isinstance(data, dict) else msg.data.strip()
        except (json.JSONDecodeError, KeyError):
            reply = msg.data.strip()

        if not reply:
            return

        preview = f'"{reply[:80]}..."' if len(reply) > 80 else f'"{reply}"'
        self.get_logger().info(f'Speaking: {preview}')
        threading.Thread(target=self._speak, args=(reply,), daemon=True).start()

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    def _speak(self, text: str):
        with self.speak_lock:
            self.is_speaking = True
            self._publish_status('speaking')

            try:
                if self.tts_mode == 'python':
                    self._speak_python(text)
                else:
                    self._speak_binary(text)
            except Exception as e:
                self.get_logger().error(f'TTS error: {e}')
                self._publish_status('error')
                self.is_speaking = False
                return

            self.is_speaking = False
            self._publish_status('idle')
            if self.continuous_mode:
                import time
                time.sleep(0.5)
                trigger = Bool()
                trigger.data = True
                self.trigger_pub.publish(trigger)
                self.get_logger().info('Continuous mode — auto-triggering next turn')

    def _speak_python(self, text: str):
        if self.voice is None:
            self.get_logger().error('No Python voice loaded')
            return

        chunks = []
        native_sr = self.voice.config.sample_rate
        for audio_chunk in self.voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)

        if not chunks:
            self.get_logger().warn('Piper produced no audio')
            return

        audio = np.concatenate(chunks).astype(np.float32)
        audio = self._resample(audio, native_sr, self.output_sample_rate)
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        sd.play(audio, self.output_sample_rate, device=self.output_device)
        sd.wait()
        self.get_logger().info('Playback complete')

    def _speak_binary(self, text: str):
        cmd = [
            '/home/unitree/unitree_sdk2_latest/build/bin/g1_piper_tts',
            'eth0',
        ]
        result = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            self.get_logger().error(f'g1_piper_tts error: {result.stderr}')
            return
        self.get_logger().info('Playback complete')

    def _continuous_start_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = True
            self.get_logger().info('Continuous mode enabled')
    
    def _continuous_stop_cb(self, msg: Bool):
        if msg.data:
            self.continuous_mode = False
            self.get_logger().info('Continuous mode disabled')
    
    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _resolve_default_output(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            # 1. First priority: look for onboard analog / ALC speakers
            for idx, dev in enumerate(devices):
                name = dev.get('name', '').lower()
                max_out = dev.get('max_output_channels', 0)
                if max_out > 0 and ('alc' in name or 'analog' in name or 'speaker' in name):
                    self.get_logger().info(f'Auto-selected audio output: [{idx}] {dev["name"]}')
                    return idx

            # 2. Second priority: system default output if it has output channels
            default_out = sd.default.device[1]
            if default_out is not None and default_out >= 0:
                if devices[default_out].get('max_output_channels', 0) > 0:
                    self.get_logger().info(f'Using system default audio output: [{default_out}] {devices[default_out]["name"]}')
                    return default_out

            # 3. Fallback: first available device with output channels > 0
            for idx, dev in enumerate(devices):
                if dev.get('max_output_channels', 0) > 0 and 'hdmi' not in dev.get('name', '').lower():
                    self.get_logger().info(f'Fallback audio output: [{idx}] {dev["name"]}')
                    return idx
        except Exception as e:
            self.get_logger().warn(f'Failed to auto-detect audio device: {e}')

        return None


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
