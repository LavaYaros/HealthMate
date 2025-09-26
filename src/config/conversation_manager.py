"""
Module for managing conversations, including default conversation handling.
Provides functionality to check for existing default conversation and create one if needed.
"""
from typing import Dict, Optional, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONV_NAME = "Default"
STORAGE_PATH = Path(__file__).parent.parent / "storage" / "conversations.json"

def ensure_storage_exists() -> None:
    """Ensure the storage directory and file exist."""
    STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STORAGE_PATH.exists():
        STORAGE_PATH.write_text('{"conversations": []}')

def load_conversations() -> Dict:
    """Load all conversations from storage."""
    ensure_storage_exists()
    try:
        return json.loads(STORAGE_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to load conversations: {e}")
        return {"conversations": []}

def save_conversations(data: Dict) -> None:
    """Save conversations to storage."""
    ensure_storage_exists()
    try:
        STORAGE_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to save conversations: {e}")

def find_default_conversation() -> Optional[Dict]:
    """Find existing default conversation in storage.
    
    Returns:
        Dict containing conversation data if found, None otherwise
    """
    data = load_conversations()
    for conv in data.get("conversations", []):
        if conv.get("title") == DEFAULT_CONV_NAME:
            return conv
    return None

def create_default_conversation(conversation_id: str) -> Dict:
    """Create a new default conversation with the given ID.
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        Dict containing the new conversation data
    """
    new_conv = {
        "conversation_id": conversation_id,
        "title": DEFAULT_CONV_NAME,
        "messages": []
    }
    
    data = load_conversations()
    data["conversations"].append(new_conv)
    save_conversations(data)
    return new_conv

def get_or_create_default_conversation(conversation_id: str) -> Tuple[str, bool]:
    """Get existing default conversation or create new one if needed.
    
    Args:
        conversation_id: ID to use if creating new conversation
        
    Returns:
        Tuple of (conversation_id, was_created)
        where was_created is True if a new conversation was created
    """
    existing = find_default_conversation()
    if existing:
        logger.info(f"Found existing default conversation: {existing['conversation_id']}")
        return existing["conversation_id"], False
        
    # No default conversation found, create new one
    new_conv = create_default_conversation(conversation_id)
    logger.info(f"Created new default conversation: {conversation_id}")
    return new_conv["conversation_id"], True