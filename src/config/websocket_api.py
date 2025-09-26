from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import re
import asyncio
import json
from src.logging.logger import setup_logger
from src.rag.pipeline import generate
from src.memory.graph_state import LangGraphStateManager

logger = setup_logger(__name__)
state_manager = LangGraphStateManager()

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint that forwards user messages to the `generate` pipeline and
    streams the assistant reply token-by-token.

    This implementation simulates streaming by splitting the final assistant
    reply into small chunks (words) and sending them with a short asyncio.sleep
    between sends. Later, if your LLM provider supports true streaming, you
    can replace the chunking logic with reading tokens directly from the
    provider and forwarding them as they arrive.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message")
            conversation_id = payload.get("conversation_id")
            
            if not message or not conversation_id:
                logger.warning("WebSocket: received incomplete payload: %s", payload)
                await websocket.send_text("__ERROR__: Missing message or conversation_id")
                continue
            
            logger.info("WebSocket: processing message for conversation_id=%s", conversation_id)
            
            try:
                # Call the generate pipeline with state management
                state = state_manager.get_state(conversation_id)
                # Use state manager's add_message method instead of direct append
                state_manager.add_message(conversation_id, "user", message)
                # Refresh state to get the updated messages
                state = state_manager.get_state(conversation_id)
                
                out_d = await generate(
                    state=state, 
                    conversation_id=conversation_id,
                    state_manager=state_manager
                )
                full_resp = out_d.get('answer') if isinstance(out_d, dict) else out_d

                if not full_resp:
                    logger.warning("WebSocket: empty response from generate pipeline")
                    await websocket.send_text("I apologize, but I couldn't generate a response.")
                    await websocket.send_text("__DONE__")
                    continue

                # extract assistant-only reply similar to the HTTP endpoint
                parts = re.split(r'(?i)assistant:\s*', full_resp)
                if len(parts) > 1:
                    assistant_reply = parts[-1].strip()
                else:
                    # If no "Assistant:" prefix found, use the full response
                    assistant_reply = full_resp.strip()

                logger.info("WebSocket: streaming response with %d characters", len(assistant_reply))

                # Simulate token-by-token streaming by sending word chunks with a tiny delay.
                # This gives front-ends (JS, Gradio, etc.) an incremental stream to display.
                tokens = re.split(r'(\s+)', assistant_reply)  # keep spaces as separate tokens
                token_count = 0
                for tok in tokens:
                    if not tok:
                        continue
                    await websocket.send_text(tok)
                    token_count += 1
                    # short pause so client can render tokens incrementally
                    await asyncio.sleep(0.03)
                
                logger.info(f"WebSocket: sent {token_count} tokens total")
                
                # send a final sentinel to indicate completion
                await websocket.send_text("__DONE__")
                logger.info("WebSocket: completed streaming for conversation_id=%s", conversation_id)
                
            except Exception as e:
                logger.exception("WebSocket: error processing message: %s", e)
                try:
                    await websocket.send_text(f"__ERROR__: {str(e)}")
                    await websocket.send_text("__DONE__")
                except RuntimeError:
                    # Connection already closed, nothing we can do
                    pass
                
    except WebSocketDisconnect:
        # client closed the connection normally (1000) — stop processing
        logger.info("WebSocket disconnected by client")
        return
    except Exception as exc:
        # log unexpected errors and close the socket if open
        logger.exception("Error in websocket_chat: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
        return
