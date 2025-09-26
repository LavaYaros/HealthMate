import sys
from pathlib import Path

# add project root (parent of "src") to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import time
import uuid
import re
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body, WebSocket
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.rag.pipeline import generate
from src.memory.graph_state import LangGraphStateManager, MessageDict
from src.logging.logger import setup_logger
from src.config.conversation_manager import get_or_create_default_conversation

logger = setup_logger(__name__)
settings = get_settings()

state_manager = LangGraphStateManager()

DEFAULT_CONV_ID, _ = get_or_create_default_conversation(str(uuid.uuid4()))
state_manager.get_state(DEFAULT_CONV_ID, title="Default")

app = FastAPI(
    title="HealthMate API",
    websocket_keepalive=20  # 20 second keepalive timeout
)

# Include websocket router
from .websocket_api import router as websocket_router
app.include_router(websocket_router)

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[MessageDict]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=512, alias="max_tokens")
    conversation_id: Optional[str] = None

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    """Process a chat completion request using LangGraph state management."""
    model_id = req.model or settings.LLM_MODEL or settings.MODEL_ID
    assert model_id, "MODEL_ID must be set"

    # Get or create conversation state
    conversation_id = req.conversation_id or DEFAULT_CONV_ID
    state = state_manager.get_state(conversation_id)
    
    # Get the last user message
    user_query = req.messages[-1].content if req.messages else ""
    
    # Generate response using LangGraph pipeline
    out_d = await generate(user_query=user_query, state=state)
    
    # Extract answer from graph output
    out = out_d.get('message') if isinstance(out_d, dict) else out_d
    
    # Update conversation state with new messages
    state["messages"].extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": out}
    ])
    state_manager.update_state(conversation_id, state)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_query.split()) if user_query else 0,
            "completion_tokens": len(out.split()),
            "total_tokens": (len(user_query.split()) if user_query else 0) + len(out.split()),
        },
    }

@app.get("/v1/memory/default_id")
def get_default_memory_id():
    """Return the server-assigned default conversation id."""
    return {"ok": True, "default_conversation_id": DEFAULT_CONV_ID}

@app.get("/v1/memory/{conversation_id}")
def get_memory(conversation_id: str):
    """Get messages for a conversation."""
    try:
        messages = state_manager.get_messages(conversation_id)
        return {"ok": True, "messages": messages}
    except Exception as e:
        logger.exception("get_memory failed for %s: %s", conversation_id, e)
        return {"ok": False, "error": str(e)}

@app.post("/v1/sessions/create")
def create_session(data: Dict[str, str] = Body(...)):
    """Create a new conversation session."""
    title = data.get("title", "").strip()
    if not title:
        return {"ok": False, "error": "Chat title required"}
    
    conversation_id = str(uuid.uuid4())
    state_manager.get_state(conversation_id, title=title)
    
    return {"ok": True, "conversation_id": conversation_id, "title": title}

@app.get("/v1/sessions/list")
def list_sessions():
    """List all active conversation sessions."""
    try:
        sessions = state_manager.list_sessions()
        return {"ok": True, "sessions": sessions}
    except Exception as e:
        logger.exception("Failed to list sessions: %s", e)
        return {"ok": False, "error": str(e)}

@app.post("/v1/memory/{conversation_id}/clear")
def clear_memory_messages(conversation_id: str):
    """Clear messages from a conversation."""
    try:
        ok = state_manager.clear_messages(conversation_id)
        return {"ok": ok}
    except Exception as e:
        logger.exception("clear_memory_messages failed for %s: %s", conversation_id, e)
        return {"ok": False, "error": str(e)}

@app.delete("/v1/memory/{conversation_id}")
@app.delete("/v1/sessions/{conversation_id}")
def delete_session(conversation_id: str):
    """Delete a session entirely."""
    try:
        # Don't allow deleting default conversation
        if conversation_id == DEFAULT_CONV_ID:
            return {"ok": False, "error": "Cannot delete default conversation"}
            
        ok = state_manager.delete_conversation(conversation_id)
        return {"ok": ok}
    except Exception as e:
        logger.exception("delete_session failed for %s: %s", conversation_id, e)
        return {"ok": False, "error": str(e)}

@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {"ok": True}