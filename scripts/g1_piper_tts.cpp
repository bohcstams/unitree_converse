#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

#include "wav.hpp"

#define CHUNK_SIZE 96000

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: g1_piper_tts [NetWorkInterface(eth0)] [optional: model_path]" << std::endl;
    return 1;
  }

  // 1. Read input text from stdin
  std::string text;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!text.empty()) {
      text += " ";
    }
    text += line;
  }

  if (text.empty()) {
    std::cerr << "[g1_piper_tts] Empty input text, exiting." << std::endl;
    return 0;
  }

  // 2. Synthesize audio with Piper
  std::string model_path = (argc >= 3) ? argv[2] : "/home/unitree/.local/share/piper/en_US-lessac-medium.onnx";
  std::ifstream model_check(model_path);
  if (!model_check.good()) {
    // If not found directly, check /root/.local/share/piper with same basename
    std::string filename = model_path.substr(model_path.find_last_of("/\\") + 1);
    model_path = "/root/.local/share/piper/" + filename;
  }

  std::ifstream final_check(model_path);
  if (!final_check.good()) {
    std::cerr << "[g1_piper_tts] Voice model not found: " << model_path << std::endl;
    return 1;
  }

  std::string safe_text;
  for (char c : text) {
    if (c == '"' || c == '\\' || c == '`' || c == '$') {
      safe_text += '\\';
    }
    safe_text += c;
  }

  std::string piper_cmd = "echo \"" + safe_text + "\" | /usr/local/bin/piper --model \"" + model_path + "\" --output_file /tmp/tts_raw.wav";
  int piper_ret = std::system(piper_cmd.c_str());
  if (piper_ret != 0) {
    std::cerr << "[g1_piper_tts] Piper synthesis failed (code: " << piper_ret << ")." << std::endl;
    return 1;
  }

  // 3. Convert raw wav to 16kHz mono 16-bit PCM wav using sox
  std::string sox_cmd = "sox /tmp/tts_raw.wav -r 16000 -c 1 -b 16 /tmp/tts_16k.wav > /dev/null 2>&1";
  int sox_ret = std::system(sox_cmd.c_str());
  if (sox_ret != 0) {
    std::cerr << "[g1_piper_tts] sox conversion failed." << std::endl;
    return 1;
  }

  // 4. Read converted wave
  int32_t sample_rate = -1;
  int8_t num_channels = 0;
  bool filestate = false;
  std::vector<uint8_t> pcm =
      ReadWave("/tmp/tts_16k.wav", &sample_rate, &num_channels, &filestate);

  if (!filestate || sample_rate != 16000 || num_channels != 1) {
    std::cerr << "[g1_piper_tts] Invalid audio format in /tmp/tts_16k.wav." << std::endl;
    return 1;
  }

  // 5. Initialize Unitree AudioClient & Stream to robot speaker
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::g1::AudioClient client;
  client.Init();
  client.SetTimeout(10.0f);

  size_t total_size = pcm.size();
  size_t offset = 0;
  std::string stream_id = std::to_string(unitree::common::GetCurrentTimeMillisecond());
  double total_duration_sec = static_cast<double>(total_size) / 32000.0;
  auto start_time = std::chrono::steady_clock::now();

  while (offset < total_size) {
    size_t remaining = total_size - offset;
    size_t current_chunk_size =
        std::min(static_cast<size_t>(CHUNK_SIZE), remaining);
    std::vector<uint8_t> chunk(pcm.begin() + offset,
                               pcm.begin() + offset + current_chunk_size);
    client.PlayStream("tts", stream_id, chunk);
    offset += current_chunk_size;
    if (offset < total_size) {
      unitree::common::MilliSleep(1000);
    }
  }

  // Wait until playback completes on robot speaker hardware (+400ms margin)
  int64_t total_duration_ms = static_cast<int64_t>(total_duration_sec * 1000.0);
  while (true) {
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    if (elapsed_ms >= total_duration_ms + 400) {
      break;
    }
    unitree::common::MilliSleep(50);
  }

  client.PlayStop("tts");
  std::cout << "[g1_piper_tts] Audio playback completed successfully." << std::endl;
  return 0;
}
