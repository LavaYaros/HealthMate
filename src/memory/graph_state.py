"""LangGraph state management for conversation sessions with single JSON storage."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from typing_extensions import Annotated

from src.config.settings import get_settings
from src.logging.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)

class MessageDict(TypedDict):
    role: str
    content: str
    message_id: str
    timestamp: str

class ConversationDict(TypedDict):
    conversation_id: str
    title: str
    messages: List[MessageDict]
    summary: str

class GraphState(TypedDict):
    messages: Annotated[List[MessageDict], "add_messages"]
    title: str
    summary: Optional[str]

class LangGraphStateManager:
    """Manages LangGraph state for multiple conversation sessions.
    
    Stores all conversations in a single conversations.json file with the format:
    {
        "conversation_id": {
            "conversation_id": str,
            "title": str,
            "messages": [
                {
                    "role": str,
                    "content": str,
                    "message_id": str,
                    "timestamp": str (ISO format)
                }
            ],
            "summary": str
        }
    }
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_dir = Path(storage_path or settings.MEMORY_PATH)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_file = self.storage_dir / "conversations.json"
        self._sessions: Dict[str, ConversationDict] = {}
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load all sessions from conversations.json."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, "r") as f:
                    self._sessions = json.load(f)
            else:
                self._sessions = {}
                self._save_sessions()
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self._sessions = {}

    def _save_sessions(self) -> None:
        """Save all sessions to conversations.json."""
        try:
            with open(self.storage_file, "w") as f:
                json.dump(self._sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def _create_message(self, role: str, content: str) -> MessageDict:
        """Create a new message with UUID and timestamp."""
        return MessageDict(
            role=role,
            content=content,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat()
        )

    def get_state(self, conversation_id: str, title: Optional[str] = None) -> GraphState:
        """Get or create a conversation state."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationDict(
                conversation_id=conversation_id,
                title=title or "Untitled",
                messages=[],
                summary=""
            )
            self._save_sessions()
        
        conversation = self._sessions[conversation_id]
        return GraphState(
            messages=conversation["messages"],
            title=conversation["title"],
            summary=conversation["summary"]
        )

    def update_state(self, conversation_id: str, state: GraphState) -> None:
        """Update a conversation's state and persist changes."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationDict(
                conversation_id=conversation_id,
                title=state["title"],
                messages=[],
                summary=state.get("summary", "")
            )
        
        # Ensure all messages have proper format with IDs and timestamps
        messages = []
        for msg in state["messages"]:
            if "message_id" not in msg:
                # Convert old message format to new format
                messages.append(self._create_message(msg["role"], msg["content"]))
            else:
                messages.append(msg)
        
        self._sessions[conversation_id].update(
            title=state["title"],
            messages=messages,
            summary=state.get("summary", "")
        )
        
        self._save_sessions()

    def clear_messages(self, conversation_id: str) -> bool:
        """Clear messages from a conversation but keep the session."""
        if conversation_id in self._sessions:
            self._sessions[conversation_id]["messages"] = []
            self._save_sessions()
            return True
        return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation entirely."""
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
            self._save_sessions()
            return True
        return False

    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        return [
            {
                "conversation_id": conv_id,
                "title": conv["title"],
                "message_count": len(conv["messages"])
            }
            for conv_id, conv in self._sessions.items()
        ]

    def get_messages(self, conversation_id: str) -> List[MessageDict]:
        """Get messages for UI display."""
        self._load_sessions()  # Reload from disk first
        conversation = self._sessions.get(conversation_id)
        return conversation["messages"] if conversation else []

    def add_message(self, conversation_id: str, role: str, content: str) -> MessageDict:
        """Add a new message to a conversation with proper ID and timestamp."""
        if conversation_id not in self._sessions:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message = self._create_message(role, content)
        self._sessions[conversation_id]["messages"].append(message)
        self._save_sessions()
        return message