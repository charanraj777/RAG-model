import uuid
from typing import Dict, List


class ConversationMemory:
    def __init__(self) -> None:
        self._store: Dict[str, List[str]] = {}

    def new_conversation(self) -> str:
        conv_id = str(uuid.uuid4())
        self._store[conv_id] = []
        return conv_id

    def append_turn(self, conversation_id: str, user: str, assistant: str) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(f"User: {user}")
        self._store[conversation_id].append(f"Assistant: {assistant}")

    def get_history(self, conversation_id: str) -> List[str]:
        return self._store.get(conversation_id, [])

