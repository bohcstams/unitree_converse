import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import threading
import time


class ActionDispatcherNode(Node):
    """
    Subscribes to 'llm_response', parses structured action plans from the LLM,
    publishes action commands to robot manipulation controllers, and triggers
    spoken voice confirmation upon action completion.
    """

    def __init__(self):
        super().__init__('action_dispatcher_node')

        self.declare_parameter('completion_phrase', 'Kész vagyok! Mi a következő kérésed?')
        self.declare_parameter('auto_listen_after_done', False)
        self.declare_parameter('simulate_execution_sec', 0.0)

        self.completion_phrase = self.get_parameter('completion_phrase').value
        self.auto_listen = self.get_parameter('auto_listen_after_done').value
        self.simulate_sec = self.get_parameter('simulate_execution_sec').value

        # Listen to LLM response containing JSON with "reply" and "actions"
        self.llm_sub = self.create_subscription(
            String, 'llm_response', self._llm_response_callback, 10)

        # Listen to execution status updates from manipulation / motion controllers
        # Controllers publish "DONE", "SUCCESS", or JSON {"status": "DONE"}
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

        self.get_logger().info('Action dispatcher node ready — listening on llm_response')

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
        self._on_action_complete()

    def _action_status_callback(self, msg: String):
        status_text = msg.data.strip().upper()
        self.get_logger().info(f'Received action status: {status_text}')

        if status_text in ('DONE', 'SUCCESS', 'COMPLETED', 'FINISHED'):
            self._on_action_complete()

    def _on_action_complete(self):
        self.is_busy = False
        self.current_plan = []

        self.get_logger().info(f'Speaking completion announcement: "{self.completion_phrase}"')
        say_msg = String()
        say_msg.data = self.completion_phrase
        self.say_pub.publish(say_msg)

        if self.auto_listen:
            # Short delay to allow completion phrase to start/finish before opening mic
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
