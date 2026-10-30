# runtime/message_bus.py
import json
import time
import uuid
from pathlib import Path


class MessageBus:
    """
    In-memory state + JSONL event log to support traceability and debugging.
    """

    def __init__(self, log_dir: Path):
        self.state: dict[str, object] = {}
        self.events: list[dict] = []
        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.events_path.touch(exist_ok=True)

    def publish(self, sender: str, payload: dict):
        event = {
            "id": str(uuid.uuid4()),
            "sender": sender,
            "timestamp": time.time(),
            "payload": payload,
        }
        self.events.append(event)

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def get_state(self, key: str, default=None):
        return self.state.get(key, default)

    def set_state(self, key: str, value):
        self.state[key] = value
