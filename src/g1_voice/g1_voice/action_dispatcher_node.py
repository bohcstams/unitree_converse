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
    'looking': {
        'hu': [
            'Máris körülnézek.',
            'Hmm, hadd nézzem meg.',
        ],
        'en': [
            'Hmm, let me take a look.',
            'Let me take a look right away.',
        ],
    },
    'accepted': {
        'hu': [
            'Rendben, máris csinálom.',
            'Értettem, máris nekilátok.',
            'Megtaláltam a tárgyat, máris csinálom.',
        ],
        'en': [
            "Alright, I'll do it right away.",
            "Got it, I'll do it right away.",
            'I see the object, doing it right away.',
        ],
    },
    'not_found': {
        'hu': [
            'Sajnos nem találom a tárgyat.',
            'Nem látom ezt a tárgyat az asztalon.',
            'Sajnos nem találom a kért tárgyat.',
        ],
        'en': [
            "I can't find the object.",
            "Sorry, I can't see that object on the table.",
            "I can't find the requested object.",
        ],
    },
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
            'All done! How else can I help?',
            'The objects are in place. What is the next task?',
            'Action finished! Ready for the next command.',
        ],
    },
    'error': {
        'hu': [
            'Sajnos nem sikerült befejezni a feladatot.',
            'Hiba történt a végrehajtás során.',
        ],
        'en': [
            'Sorry, I was unable to complete the task.',
            'An error occurred while executing the action.',
        ],
    },
}

# Generic placement surfaces that do not require a specific YOLO object bounding box
SURFACE_TARGETS = {'table', 'asztal', 'desk', 'surface'}

# Actions that ask the robot to describe what objects it sees on the table
LIST_OBJECT_ACTIONS = {'list_objects', 'look', 'describe', 'scan'}

# Canonical object aliases so e.g. "box" matches "black_box" and "cup"/"glass" matches "glass_cup"
TARGET_ALIASES = {
    'box': ['black_box', 'box'],
    'black_box': ['black_box', 'box'],
    'cup': ['glass_cup', 'cup', 'mug'],
    'glass': ['glass_cup', 'cup'],
    'glass_cup': ['glass_cup', 'cup'],
    'mug': ['mug', 'cup'],
    'pen': ['pen'],
    'cube': ['rubiks_cube', 'red_cube', 'blue_cube'],
    'rubiks_cube': ['rubiks_cube'],
    'sunglasses': ['sunglasses', 'glasses'],
    'paddle': ['ping_pong_paddle'],
    'ping_pong_paddle': ['ping_pong_paddle'],
    'headset': ['headset', 'headphones'],
    'bottle': ['bottle'],
    'phone': ['phone', 'cell_phone'],
}


class ActionDispatcherNode(Node):
    """
    Subscribes to:
      - 'llm_response': structured JSON action plans from the LLM
      - '/g1/vision/detections': latest detected objects & 2D bounding boxes from g1_vision
      - '/g1/action/status': completion/error status from the motion planner

    When an action command arrives:
      1. If action is 'list_objects', speaks the initial look-around reply (e.g.
         "Hmm, let me take a look" / "Máris körülnézek") and sends the currently
         detected English object names to the LLM via '[VISION]: <objects>' so the
         LLM generates a natural spoken sentence listing the visible objects.
      2. Otherwise, looks up the requested target object(s) in the latest vision detections.
      3. If a required object is not visible, speaks "I can't find the object" /
         "Sajnos nem találom a tárgyat" and aborts dispatch.
      4. If found, speaks "Alright, I'll do it right away" / "Rendben, máris csinálom",
         enriches each action item with its 2D bounding box / vision metadata, and
         publishes the enriched plan to '/g1/action/commands' for the motion planner.
    """

    def __init__(self):
        super().__init__('action_dispatcher_node')

        try:
            pkg_share = get_package_share_directory('g1_voice')
            default_phrases_file = os.path.join(pkg_share, 'config', 'feedback_phrases.yaml')
        except Exception:
            default_phrases_file = ''

        self.declare_parameter('language', 'hu')
        self.declare_parameter('phrases_file', default_phrases_file)
        self.declare_parameter('detections_topic', '/g1/vision/detections')
        self.declare_parameter('llm_prompt_topic', '/g1/stt/transcript')
        self.declare_parameter('auto_listen_after_done', False)
        self.declare_parameter('simulate_execution_sec', 0.0)

        self.language = self.get_parameter('language').value
        self.phrases_file = self.get_parameter('phrases_file').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.llm_prompt_topic = self.get_parameter('llm_prompt_topic').value
        self.auto_listen = self.get_parameter('auto_listen_after_done').value
        self.simulate_sec = self.get_parameter('simulate_execution_sec').value

        # Load response phrases pool
        self.phrases = self._load_phrases()

        # Latest vision detections state (thread-safe)
        self._vision_lock = threading.Lock()
        self._latest_detections = []
        self._latest_frame_id = 'camera_color_optical_frame'
        self._latest_image_size = {'width': 640, 'height': 480}
        self._latest_vision_timestamp = None
        self._awaiting_vision_reply = False

        # Subscribe to vision detections from g1_vision
        self.vision_sub = self.create_subscription(
            String, self.detections_topic, self._vision_detections_callback, 10
        )

        # Listen to LLM response containing JSON with "reply" and "actions"
        self.llm_sub = self.create_subscription(
            String, 'llm_response', self._llm_response_callback, 10
        )

        # Listen to execution status updates from manipulation / motion controllers
        self.status_sub = self.create_subscription(
            String, '/g1/action/status', self._action_status_callback, 10
        )

        # Publish enriched action list to motion planner / arm controllers
        self.commands_pub = self.create_publisher(String, '/g1/action/commands', 10)

        # Publish speech requests to TTS (speaks through robot head speaker)
        self.say_pub = self.create_publisher(String, '/g1/tts/say', 10)

        # Publish follow-up [VISION]: prompts to the LLM node
        self.llm_prompt_pub = self.create_publisher(String, self.llm_prompt_topic, 10)

        # Optional trigger to auto-activate microphone for next turn
        self.trigger_pub = self.create_publisher(Bool, '/g1/voice/trigger', 10)

        self.is_busy = False
        self.current_plan = []

        self.get_logger().info(
            f'Action dispatcher node ready — language: {self.language}, '
            f'listening on llm_response and {self.detections_topic}'
        )

    def _load_phrases(self) -> dict:
        """Load feedback phrases from YAML file, merging with defaults."""
        candidates = []
        if self.phrases_file:
            candidates.append(self.phrases_file)

        src_path = '/root/ros2_ws/src/unitree_converse/src/g1_voice/config/feedback_phrases.yaml'
        candidates.append(src_path)

        local_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'feedback_phrases.yaml')
        candidates.append(local_path)

        merged = dict(DEFAULT_FALLBACK_PHRASES)
        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        merged.update(data)
                        self.get_logger().info(f'Loaded feedback phrases from: {path}')
                        return merged
                except Exception as e:
                    self.get_logger().warn(f'Failed to parse phrases file {path}: {e}')

        self.get_logger().info('Using built-in default feedback phrases')
        return merged

    def _get_random_phrase(self, category: str) -> str:
        """Pick a random phrase for the category and configured language."""
        lang_dict = self.phrases.get(category, {})
        pool = lang_dict.get(self.language)

        if not pool:
            pool = lang_dict.get('en') or next(iter(lang_dict.values()), None)

        if pool and isinstance(pool, list):
            return random.choice(pool)

        return DEFAULT_FALLBACK_PHRASES.get(category, {}).get(self.language, ["Alright."])[0]

    def _speak_text(self, text: str):
        """Publish an explicit sentence to /g1/tts/say."""
        self.get_logger().info(f'Speaking ({self.language}): "{text}"')
        say_msg = String()
        say_msg.data = text
        self.say_pub.publish(say_msg)

    def _speak_phrase(self, category: str):
        """Select a random phrase from category and publish it to /g1/tts/say."""
        phrase = self._get_random_phrase(category)
        self.get_logger().info(f'Speaking ({category}, {self.language}): "{phrase}"')
        say_msg = String()
        say_msg.data = phrase
        self.say_pub.publish(say_msg)

    def _handle_list_objects(self):
        """Collect unique detected object names and send a [VISION]: prompt to the LLM."""
        with self._vision_lock:
            detections = list(self._latest_detections)

        seen_names = []
        sorted_dets = sorted(
            detections,
            key=lambda d: float(d.get('confidence', 0.0)) if isinstance(d, dict) else 0.0,
            reverse=True,
        )
        for det in sorted_dets:
            if not isinstance(det, dict):
                continue
            name = str(det.get('name', '')).strip()
            if name and name not in seen_names:
                seen_names.append(name)

        if seen_names:
            vision_query = f"[VISION]: {', '.join(seen_names)}"
        else:
            vision_query = '[VISION]: none'

        self._awaiting_vision_reply = True
        prompt_msg = String()
        prompt_msg.data = vision_query
        self.llm_prompt_pub.publish(prompt_msg)
        self.get_logger().info(
            f'Sent detected objects to LLM ({self.llm_prompt_topic}): "{vision_query}"'
        )

    def _vision_detections_callback(self, msg: String):
        """Store the latest object detections from /g1/vision/detections."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        detections = data.get('detections', [])
        if not isinstance(detections, list):
            return

        with self._vision_lock:
            self._latest_detections = detections
            self._latest_frame_id = data.get('frame_id', 'camera_color_optical_frame')
            self._latest_image_size = data.get('image_size', {'width': 640, 'height': 480})
            self._latest_vision_timestamp = data.get('timestamp')

    def _find_object_in_detections(self, target_name: str):
        """
        Find the best (highest-confidence) detection matching `target_name`
        in the latest vision detections list.
        """
        norm_target = target_name.strip().lower().replace(' ', '_')
        candidate_names = TARGET_ALIASES.get(norm_target, [norm_target])

        with self._vision_lock:
            detections = list(self._latest_detections)
            frame_id = self._latest_frame_id
            image_size = dict(self._latest_image_size)
            timestamp = self._latest_vision_timestamp

        # 1. Check exact / alias match in priority order
        for candidate in candidate_names:
            matches = [
                d for d in detections
                if isinstance(d, dict) and str(d.get('name', '')).strip().lower().replace(' ', '_') == candidate
            ]
            if matches:
                best = max(matches, key=lambda d: float(d.get('confidence', 0.0)))
                return best, frame_id, image_size, timestamp

        # 2. Fallback: substring match (e.g. "box" in "black_box")
        substring_matches = [
            d for d in detections
            if isinstance(d, dict) and (
                norm_target in str(d.get('name', '')).strip().lower().replace(' ', '_')
                or str(d.get('name', '')).strip().lower().replace(' ', '_') in norm_target
            )
        ]
        if substring_matches:
            best = max(substring_matches, key=lambda d: float(d.get('confidence', 0.0)))
            return best, frame_id, image_size, timestamp

        return None, frame_id, image_size, timestamp

    def _llm_response_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().debug('llm_response is not JSON; ignoring for action dispatch')
            return

        if not isinstance(data, dict):
            return

        actions = data.get('actions', [])

        # Safeguard: if this response is the LLM's reply to our [VISION]: query,
        # never trigger another action loop. If the LLM accidentally returned
        # non-empty actions (meaning tts_node deferred speaking), speak the reply here.
        if self._awaiting_vision_reply:
            self._awaiting_vision_reply = False
            if isinstance(actions, list) and len(actions) > 0:
                reply_text = str(data.get('reply', '')).strip()
                if reply_text:
                    self._speak_text(reply_text)
            return

        if not actions or not isinstance(actions, list):
            self.get_logger().info('No actions requested in this turn (conversational turn).')
            return

        self.get_logger().info(f'Parsed {len(actions)} action(s) from LLM: {actions}')

        # Check if the user asked what objects the robot sees ('list_objects')
        if any(
            isinstance(s, dict) and str(s.get('action', '')).strip().lower() in LIST_OBJECT_ACTIONS
            for s in actions
        ):
            reply_text = str(data.get('reply', '')).strip()
            if reply_text:
                self._speak_text(reply_text)
            else:
                self._speak_phrase('looking')
            self._handle_list_objects()
            return

        enriched_actions = []
        for step in actions:
            if not isinstance(step, dict):
                continue
            action_type = str(step.get('action', '')).strip().lower()
            target = str(step.get('target', '')).strip()
            norm_target = target.lower().replace(' ', '_')

            enriched_step = dict(step)

            # Generic surfaces like "table" do not require a YOLO object bounding box
            if not norm_target or norm_target in SURFACE_TARGETS:
                enriched_step['target_type'] = 'surface'
                enriched_actions.append(enriched_step)
                continue

            matched_det, frame_id, image_size, timestamp = self._find_object_in_detections(norm_target)
            if matched_det is None:
                with self._vision_lock:
                    visible_names = [d.get('name') for d in self._latest_detections if isinstance(d, dict)]
                self.get_logger().warn(
                    f'Requested target "{target}" for action "{action_type}" not found in vision detections '
                    f'(currently visible: {visible_names}).'
                )
                self._speak_phrase('not_found')
                return

            enriched_step['target_type'] = 'object'
            enriched_step['detection'] = matched_det
            enriched_step['frame_id'] = frame_id
            enriched_step['image_size'] = image_size
            if timestamp is not None:
                enriched_step['vision_timestamp'] = timestamp
            enriched_actions.append(enriched_step)

        if not enriched_actions:
            return

        # All required objects were found! Speak confirmation ("Alright, I'll do it right away")
        self._speak_phrase('accepted')

        self.current_plan = enriched_actions
        self.is_busy = True

        # Forward enriched action plan (with bounding boxes & coordinates) to motion planner
        cmd_msg = String()
        cmd_msg.data = json.dumps(enriched_actions, ensure_ascii=False)
        self.commands_pub.publish(cmd_msg)
        self.get_logger().info(f'Dispatched enriched action plan to /g1/action/commands: {cmd_msg.data}')

        # If simulated execution is requested (e.g. testing without physical arm controller):
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
        self._speak_phrase(category)

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
