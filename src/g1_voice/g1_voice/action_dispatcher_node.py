import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from ament_index_python.packages import get_package_share_directory
import json
import threading
import random
import time
import os
import yaml


DEFAULT_FALLBACK_PHRASES = {
    'completion': {
        'hu': [
            'Kész vagyok! Mi a következő kérésed?',
            'Befejeztem a feladatot! Miben segíthetek még?',
            'A tárgyakat elrendeztem. Mi legyen a következő lépés?',
            'Kész a művelet! Várom a következő utasítást.',
            'Sikeresen végeztem! Mi a következő feladat?',
        ],
        'en': [
            "I'm done! What's the next request?",
            "I've completed the task. What should I do next?",
            "All done! How else can I help?",
            "The objects are in place. What is the next task?",
            "Action finished! Ready for the next command.",
        ]
    },
    'error': {
        'hu': [
            'Sajnos nem sikerült befejezni a feladatot.',
            'Hiba történt a végrehajtás során.',
        ],
        'en': [
            'Sorry, I was unable to complete the task.',
            'An error occurred while executing the action.',
        ]
    }
}


class ActionDispatcherNode(Node):
    """
    Subscribes to 'llm_response', parses structured action plans from the LLM,
    publishes action commands to robot manipulation controllers, and triggers
    spoken voice confirmation upon action completion in the configured language.
    """

    def __init__(self):
        super().__init__('action_dispatcher_node')

        # Determine default phrases file location
        try:
            pkg_share = get_package_share_directory('g1_voice')
            default_phrases_file = os.path.join(pkg_share, 'config', 'feedback_phrases.yaml')
        except Exception:
            default_phrases_file = ''

        self.declare_parameter('language', 'hu')
        self.declare_parameter('phrases_file', default_phrases_file)
        self.declare_parameter('auto_listen_after_done', False)
        self.declare_parameter('simulate_execution_sec', 0.0)

        self.language = self.get_parameter('language').value
        self.phrases_file = self.get_parameter('phrases_file').value
        self.auto_listen = self.get_parameter('auto_listen_after_done').value
        self.simulate_sec = self.get_parameter('simulate_execution_sec').value

        # Load response phrases pool
        self.phrases = self._load_phrases()

        # Listen to LLM response containing JSON with "reply" and "actions"
        self.llm_sub = self.create_subscription(
            String, 'llm_response', self._llm_response_callback, 10)

        # Listen to execution status updates from manipulation / motion controllers
        # Controllers publish "DONE", "SUCCESS", "ERROR", etc.
        self.status_sub = self.create_subscription(
            String, '/g1/action/status', self._action_status_callback, 10)

        # Publish parsed action list to motion planner / arm controllers
        self.commands_pub = self.create_publisher(String, '/g1/action/commands', 10)

        # Publish speech requests to TTS (speaks through robot head speaker)
        self.say_pub = self.create_publisher(String, '/g1/tts/say', 10)

        # Optional trigger to auto-activate microphone for next turn
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)

        self.is_busy = False
        self.current_plan = []

        self.get_logger().info(
            f'Action dispatcher node ready — language: {self.language}, listening on llm_response')

    def _load_phrases(self) -> dict:
        """Load feedback phrases from YAML file, with automatic fallback."""
        candidates = []
        if self.phrases_file:
            candidates.append(self.phrases_file)

        # Also check source directory path in case of container mount
        src_path = '/root/ros2_ws/src/unitree_converse/src/g1_voice/config/feedback_phrases.yaml'
        candidates.append(src_path)

        local_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'feedback_phrases.yaml')
        candidates.append(local_path)

        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        self.get_logger().info(f'Loaded feedback phrases from: {path}')
                        return data
                except Exception as e:
                    self.get_logger().warn(f'Failed to parse phrases file {path}: {e}')

        self.get_logger().info('Using built-in default feedback phrases')
        return DEFAULT_FALLBACK_PHRASES

    def _get_random_phrase(self, category: str) -> str:
        """Pick a random phrase for the category and configured language."""
        lang_dict = self.phrases.get(category, {})
        pool = lang_dict.get(self.language)

        if not pool:
            # Fallback to English, then to any available language in the category
            pool = lang_dict.get('en') or next(iter(lang_dict.values()), None)

        if pool and isinstance(pool, list):
            return random.choice(pool)

        return "I'm done! What's the next request?"

    def _llm_response_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().debug('llm_response is not JSON; ignoring for action dispatch')
            return

        if not isinstance(data, dict):
            return

        actions = data.get('actions', [])
        if not actions or not isinstance(actions, list):
            self.get_logger().info('No actions requested in this turn (conversational turn).')
            return

        self.get_logger().info(f'Parsed {len(actions)} action(s) from LLM: {actions}')
        self.current_plan = actions
        self.is_busy = True

        # Forward formatted action commands to robot motion controller
        cmd_msg = String()
        cmd_msg.data = json.dumps(actions, ensure_ascii=False)
        self.commands_pub.publish(cmd_msg)

        # If simulated execution is requested (e.g. testing without physical arm hardware):
        if self.simulate_sec > 0:
            threading.Thread(target=self._simulate_execution, args=(self.simulate_sec,), daemon=True).start()

    def _simulate_execution(self, duration_sec: float):
        self.get_logger().info(f'Simulating action execution ({duration_sec}s)...')
        time.sleep(duration_sec)
        self.get_logger().info('Simulated execution finished')
        self._on_action_complete(success=True)

    def _action_status_callback(self, msg: String):
        status_text = msg.data.strip().upper()
        self.get_logger().info(f'Received action status: {status_text}')

        if status_text in ('DONE', 'SUCCESS', 'COMPLETED', 'FINISHED'):
            self._on_action_complete(success=True)
        elif status_text in ('ERROR', 'FAILED', 'FAILURE'):
            self._on_action_complete(success=False)

    def _on_action_complete(self, success: bool = True):
        self.is_busy = False
        self.current_plan = []

        category = 'completion' if success else 'error'
        phrase = self._get_random_phrase(category)

        self.get_logger().info(f'Speaking completion announcement ({self.language}): "{phrase}"')
        say_msg = String()
        say_msg.data = phrase
        self.say_pub.publish(say_msg)

        if self.auto_listen and success:
            threading.Thread(target=self._delayed_trigger, args=(3.0,), daemon=True).start()

    def _delayed_trigger(self, delay_sec: float):
        time.sleep(delay_sec)
        trigger = Bool()
        trigger.data = True
        self.trigger_pub.publish(trigger)
        self.get_logger().info('Triggered microphone for next request')


def main(args=None):
    rclpy.init(args=args)
    node = ActionDispatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
