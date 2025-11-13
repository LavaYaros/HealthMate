import sys
from pathlib import Path

# add project root (parent of "src") to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import asyncio
import requests
import websockets
import gradio as gr
from functools import wraps
import base64
import tempfile

from src.audio.stt import get_stt
from src.logging.logger import setup_logger

logger = setup_logger(__name__)

stt = get_stt()

# Use environment variable for API endpoint
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_BASE = f"http://{API_HOST}:8000"
FASTAPI_ENDPOINT = f"{API_BASE}/v1/chat/completions"

def messages_to_gradio_history(messages):
    """
    Convert LangGraph message format to Gradio chat history.
    """
    history = []
    
    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            logger.warning(f"UI: invalid message format: {msg}")
            continue
            
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            # Show system messages as assistant messages
            history.append({
                'role': 'assistant',
                'content': f"[System] {content}"
            })
        elif role in ['user', 'assistant']:
            # Pass through user and assistant messages as-is
            history.append({
                'role': role,
                'content': content
            })
        else:
            logger.warning(f"UI: unknown message role: {role}")
            
    return history


async def submit_stream(message, chat_hist, sessions, active, chat_ids):
    """Async generator for Gradio that streams assistant tokens via WebSocket.
    Handles LangGraph state format and streams tokens to UI."""
    
    # ensure defaults and get conversation id
    conv_id = (chat_ids or {}).get(active, active)
    
    # Get initial state for showing the user message right away
    current_session = []
    try:
        r = requests.get(f"{API_BASE}/v1/memory/{conv_id}", timeout=5)
        if r.ok:
            current_session = r.json().get("messages", [])
    except Exception as e:
        logger.exception(f"UI: error loading initial state: {str(e)}")

    # Show user message while waiting for response
    preview_session = current_session + [{"role": "user", "content": message}]
    yield messages_to_gradio_history(preview_session), "", sessions, None

    try:
        # Connect and send message
        ws_base = API_BASE.replace('http://', 'ws://')
        uri = f"{ws_base}/ws/chat"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "message": message,
                "conversation_id": conv_id
            }))
            
            # Stream response
            response_so_far = ""
            audio_path = None
            
            while True:
                tok = await ws.recv()
                
                if tok == "__DONE__":
                    # Get final state from backend after streaming completes
                    try:
                        r = requests.get(f"{API_BASE}/v1/memory/{conv_id}", timeout=5)
                        if r.ok:
                            final_state = r.json().get("messages", [])
                            last_message_complete = final_state[-1]['content'] == response_so_far  # Ensure last message is complete
                            if last_message_complete:
                                yield messages_to_gradio_history(final_state), "", sessions, audio_path
                            else:
                                logger.warning(f"UI: final message content mismatch, using streamed content")
                    except Exception as e:
                        logger.error(f"UI: failed to get final state: {e}")
                    break
                
                if tok.startswith("__AUDIO__:"):
                    # Extract and decode audio data
                    try:
                        audio_base64 = tok.replace("__AUDIO__:", "")
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        # Save to temporary file
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        temp_audio.write(audio_bytes)
                        temp_audio.close()
                        audio_path = temp_audio.name
                        
                        logger.info(f"UI: received audio ({len(audio_bytes)} bytes), saved to {audio_path}")
                        logger.info(f"UI: audio file exists: {os.path.exists(audio_path)}")
                        logger.info(f"UI: audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'}")
                    except Exception as e:
                        logger.error(f"UI: failed to decode audio: {e}")
                    continue
                    
                if tok.startswith("__ERROR__:"):
                    error_msg = tok.replace("__ERROR__:", "").strip()
                    logger.error(f"UI: received error: {error_msg}")
                    yield messages_to_gradio_history(preview_session + [
                        {"role": "assistant", "content": f"[Error: {error_msg}]"}
                    ]), "", sessions, None
                    return

                # Show streaming progress
                response_so_far += tok
                yield messages_to_gradio_history(preview_session + [
                    {"role": "assistant", "content": response_so_far}
                ]), "", sessions, audio_path
                
    except Exception as e:
        logger.exception("UI: WebSocket error: %s", e)
        yield messages_to_gradio_history(preview_session + [
            {"role": "assistant", "content": f"[Connection Error: {str(e)}]"}
        ]), "", sessions, None


def load_chat_history(conversation_id):
    """Load and format chat history from the LangGraph backend.
    
    Args:
        conversation_id: ID of the conversation to load
        
    Returns:
        List of (user_msg, assistant_msg) tuples for Gradio chatbot
    """
    try:
        r = requests.get(f"{API_BASE}/v1/memory/{conversation_id}", timeout=5)
        if not r.ok:
            logger.error(f"UI: failed to load chat history, status {r.status_code}")
            return []
            
        data = r.json()
        if not data.get("ok"):
            logger.error(f"UI: backend returned error: {data.get('error')}")
            return []
        
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            logger.error(f"UI: invalid messages format from API: {messages}")
            return []
            
        # Ensure each message has required fields
        valid_messages = []
        for msg in messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                valid_messages.append(msg)
            else:
                logger.warning(f"UI: skipping invalid message: {msg}")
                
        return messages_to_gradio_history(valid_messages)
        
    except Exception as e:
        logger.exception(f"UI: error loading chat history for {conversation_id}: {str(e)}")
        return []

def load_existing_sessions():
    """Load existing sessions from backend at startup.
    Ensures default session is always present."""
    chat_titles = {}
    chat_ids = {}
    sessions = {}
    
    try:
        # First get the default conversation ID
        r = requests.get(f"{API_BASE}/v1/memory/default_id", timeout=5)
        if not r.ok:
            logger.error("UI: failed to get default conversation ID")
            return {}, {}, {}
            
        data = r.json()
        default_id = data.get('default_conversation_id')
        if not default_id:
            logger.error("UI: no default conversation ID returned")
            return {}, {}, {}
            
        # Load all sessions
        r = requests.get(f"{API_BASE}/v1/sessions/list", timeout=5)
        if r.ok:
            sessions_data = r.json().get('sessions', [])
            
            # First pass - load all non-default sessions
            for session in sessions_data:
                conv_id = session['conversation_id']
                title = session['title']
                
                # Skip default conversation for now
                if conv_id == default_id:
                    continue
                    
                chat_titles[conv_id] = title
                chat_ids[conv_id] = conv_id
                sessions[conv_id] = load_chat_history(conv_id)
            
            # Always ensure default session is loaded last
            default_history = load_chat_history(default_id)
            if default_history:  # Only add if we got history
                chat_titles[default_id] = "Default"
                chat_ids[default_id] = default_id
                sessions[default_id] = default_history
            
            loaded = len(sessions)
            default_status = "including" if default_id in sessions else "without"
            logger.info(f"UI: loaded {loaded} sessions ({default_status} default)")
            return chat_titles, chat_ids, sessions
            
        logger.error("UI: failed to list sessions")
        return {}, {}, {}
        
    except Exception as e:
        logger.exception("UI: error loading existing sessions: %s", e)
        return {}, {}, {}

def switch_chat_and_set_active(sessions, chat_titles, chat_ids, selected_title):
    """Switch to a chat session by title and load its history"""
    sessions = dict(sessions or {})
    chat_titles = dict(chat_titles or {})
    chat_ids = dict(chat_ids or {})
    
    # Find conversation_id by title
    conversation_id = None
    for conv_id, title in chat_titles.items():
        if title == selected_title:
            conversation_id = conv_id
            break
    
    if not conversation_id:
        logger.warning(f"UI: could not find conversation_id for title: {selected_title}")
        return [], conversation_id
    
    # Always load fresh data from backend to ensure we have latest messages
    chat_history = load_chat_history(conversation_id)
    sessions[conversation_id] = chat_history
    
    logger.info(f"UI: switched to chat {selected_title} (conv_id={conversation_id}) with {len(chat_history)} messages")
    return chat_history, conversation_id

def delete_chat(sessions, active, chat_ids, chat_titles):
    """Delete a chat session both locally and from backend"""
    sessions = dict(sessions or {})
    chat_ids = dict(chat_ids or {})
    chat_titles = dict(chat_titles or {})
    
    # Get conversation id
    conv_id = chat_ids.get(active, active)
    
    # Try to delete from backend
    try:
        r = requests.delete(f"{API_BASE}/v1/memory/{conv_id}", timeout=5)
        if not r.ok:
            error_data = r.json()
            error_msg = error_data.get('error', f"Server error: {r.status_code}")
            logger.warning(f"UI: backend delete failed: {error_msg}")
            gr.Warning(error_msg)
            # Return current state unchanged
            current_title = chat_titles.get(active, "Default")
            return (sessions, active, 
                    gr.update(choices=list(chat_titles.values()), value=current_title), 
                    sessions.get(active, []), "", chat_ids, chat_titles)
    except Exception as e:
        logger.exception("UI: backend delete failed: %s", e)
        gr.Error(f"Failed to delete conversation: {str(e)}")
        # Return current state unchanged
        current_title = chat_titles.get(active, "Default")
        return (sessions, active, 
                gr.update(choices=list(chat_titles.values()), value=current_title), 
                sessions.get(active, []), "", chat_ids, chat_titles)
    
    # Backend delete succeeded, update local state
    if active in sessions:
        del sessions[active]
    if active in chat_ids:
        del chat_ids[active]
    if active in chat_titles:
        del chat_titles[active]
    
    logger.info(f"UI: deleted session {active}")
    
    # Switch to default or first available session
    new_active = default_conv_id if default_conv_id in sessions else next(iter(sessions))
    new_title = chat_titles.get(new_active, "Default")
    
    return (sessions, new_active, 
            gr.update(choices=list(chat_titles.values()), value=new_title), 
            sessions.get(new_active, []), "", chat_ids, chat_titles)

def clear_chat(sessions, active, chat_ids, chat_titles):
    """Clear chat history both locally and on backend."""
    sessions = dict(sessions or {})
    chat_ids = dict(chat_ids or {})
    chat_titles = dict(chat_titles or {})
    
    # resolve conversation id
    conv_id = chat_ids.get(active, active)
    
    try:
        r = requests.post(f"{API_BASE}/v1/memory/{conv_id}/clear", timeout=5)
        logger.info("UI: backend clear response %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("UI: backend clear failed: %s", e)
    
    # Clear local session
    sessions[active] = []
    
    # Update dropdown
    current_title = chat_titles.get(active, "Unknown")
    dropdown_update = gr.update(choices=list(chat_titles.values()), value=current_title)
    
    return [], "", sessions, chat_ids, dropdown_update

def new_chat(chat_titles, new_chat_name, active_conv_id, sessions, chat_ids):
    """Create a new chat session."""
    chat_titles = dict(chat_titles or {})
    sessions = dict(sessions or {})
    chat_ids = dict(chat_ids or {})
    
    name = (new_chat_name or "").strip()
    if not name:
        gr.Warning("Chat name cannot be empty")
        return chat_titles, active_conv_id, gr.update(), sessions.get(active_conv_id, []), new_chat_name, sessions, chat_ids

    try:
        r = requests.post(f"{API_BASE}/v1/sessions/create", 
                         json={"title": name}, timeout=5)
        if r.ok:
            response = r.json()
            if response.get("ok"):
                conv_id = response["conversation_id"]
                chat_titles[conv_id] = name
                chat_ids[conv_id] = conv_id
                sessions[conv_id] = []
                
                return (chat_titles, conv_id,
                        gr.update(choices=list(chat_titles.values()), value=name),
                        [], "", sessions, chat_ids)
    except Exception as e:
        logger.exception("UI: failed to create new chat: %s", e)
        gr.Error(f"Failed to create chat: {str(e)}")
    
    current_sessions = sessions.get(active_conv_id, [])
    return chat_titles, active_conv_id, gr.update(), current_sessions, new_chat_name, sessions, chat_ids

def handle_audio_input(audio_path):
    if audio_path:
        return stt.transcribe(audio_path)
    return ""

async def handle_audio_and_submit(audio_path, chatbot, sessions, active, chat_ids):
    if audio_path:
        transcribed = handle_audio_input(audio_path)
        if transcribed:
            async for result in submit_stream(transcribed, chatbot, sessions, active, chat_ids):
                yield result
        else:
            yield chatbot, "", sessions, None
    else:
        yield chatbot, "", sessions, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# HealthMate - Your AI First Aid Assistant
    *I am here to guide you through home-based first aid situations. Note: Always call emergency services for life-threatening conditions.*
    """)

    # Get default conversation id from backend
    try:
        r = requests.get(f"{API_BASE}/v1/memory/default_id", timeout=2)
        if r.ok:
            default_conv = r.json().get('default_conversation_id')
        else:
            raise Exception(f"Failed to get default conversation ID: {r.status_code}")
    except Exception as e:
        logger.exception("UI: backend fetch default conversation id failed: %s", e)
        raise

    default_conv_id = default_conv
    existing_titles, existing_ids, existing_sessions = load_existing_sessions()
    
    # Initialize with default session + existing sessions
    try:
        if not existing_titles:
            initial_titles = {default_conv_id: "Default"}
            initial_ids = {default_conv_id: default_conv_id}
            initial_sessions = {default_conv_id: load_chat_history(default_conv_id)}
            dropdown_choices = ["Default"]
            dropdown_value = "Default"
        else:
            if default_conv_id not in existing_titles:
                existing_titles[default_conv_id] = "Default"
                existing_ids[default_conv_id] = default_conv_id
                existing_sessions[default_conv_id] = load_chat_history(default_conv_id)
            
            initial_titles = existing_titles
            initial_ids = existing_ids
            initial_sessions = existing_sessions
            dropdown_choices = list(initial_titles.values())
            dropdown_value = initial_titles.get(default_conv_id, dropdown_choices[0] if dropdown_choices else "Default")
        
        logger.info("UI: Successfully initialized sessions")
    except Exception as e:
        logger.exception("UI: Error initializing sessions: %s", e)
        # Provide fallback values
        initial_titles = {default_conv_id: "Default"}
        initial_ids = {default_conv_id: default_conv_id}
        initial_sessions = {default_conv_id: []}
        dropdown_choices = ["Default"]
        dropdown_value = "Default"

    # State management
    sessions = gr.State(initial_sessions)
    active = gr.State(default_conv_id)
    chat_ids = gr.State(initial_ids)
    chat_titles = gr.State(initial_titles)
    active_conv_id = gr.State(default_conv_id)
    
    # Get fresh chat history for the default session
    fresh_chat_history = load_chat_history(default_conv_id)

    with gr.Row():
        with gr.Column(scale=0, min_width=300):
            gr.Markdown("### First Aid Conversations")
            session_dd = gr.Dropdown(
                choices=dropdown_choices, 
                value=dropdown_value, 
                label="Active Case"
            )
            new_chat_name = gr.Textbox(
                placeholder="Name for this first aid case", 
                label="New Case Name"
            )
            new_btn = gr.Button("➕ New Case", variant="primary")
            clear_btn = gr.Button("🧹 Clear History")
            del_btn = gr.Button("🗑️ Delete Case", variant="stop")

        with gr.Column(scale=1):
            gr.Markdown("""### HealthMate Assistant
            *Describe your first aid situation, and I'll guide you through the appropriate steps.*""")
            
            chatbot = gr.Chatbot(
                height=500, 
                show_label=False,
                type="messages",
                avatar_images=["👤", "🩺"],
                value=fresh_chat_history
            )
            
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe the first aid situation...",
                    lines=2,
                    scale=10,
                    container=False
                )
                send = gr.Button(
                    "Send",
                    scale=1,
                    min_width=100,
                    variant="primary"
                )
            
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
            
            # Audio output for TTS responses
            audio_output = gr.Audio(
                label="🔊 Voice Response",
                autoplay=True,
                visible=True,
                show_label=True
            )

    # Wire up the components
    txt.submit(submit_stream, [txt, chatbot, sessions, active, chat_ids], [chatbot, txt, sessions, audio_output])
    send.click(submit_stream, [txt, chatbot, sessions, active, chat_ids], [chatbot, txt, sessions, audio_output])

    new_btn.click(new_chat, 
                  [chat_titles, new_chat_name, active_conv_id, sessions, chat_ids], 
                  [chat_titles, active_conv_id, session_dd, chatbot, new_chat_name, sessions, chat_ids])
    
    session_dd.change(switch_chat_and_set_active, 
                     [sessions, chat_titles, chat_ids, session_dd], 
                     [chatbot, active])
                     
    clear_btn.click(clear_chat, 
                   [sessions, active, chat_ids, chat_titles], 
                   [chatbot, txt, sessions, chat_ids, session_dd])
    
    del_btn.click(delete_chat, 
                 [sessions, active, chat_ids, chat_titles], 
                 [sessions, active, session_dd, chatbot, txt, chat_ids, chat_titles])
    
    # Add load event to refresh chat history on page load/reload
    def refresh_on_load(active_conv_id):
        """Refresh the current session on page load to get latest messages"""
        fresh_history = load_chat_history(active_conv_id)
        logger.info(f"UI: refreshed on load, loaded {len(fresh_history)} messages for {active_conv_id}")
        return fresh_history
    
    demo.load(refresh_on_load, 
              [active_conv_id], 
              [chatbot])

    audio_input.change(handle_audio_and_submit, 
                       [audio_input, chatbot, sessions, active, chat_ids], 
                       [chatbot, txt, sessions, audio_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)