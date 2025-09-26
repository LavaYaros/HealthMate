"""
LLM Adapter for HealthMate: Configurable interface to the selected LLM provider/model.
Uses LangGraph for orchestration. Prompts are scoped to home first aid, with citation and safety rules.
"""

import sys
from pathlib import Path

# add project root (parent of "src") to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import get_settings
from src.prompts.system_prompts import INSTRUCTOR_PROMPT, CHATTER_PROMPT
from src.memory.graph_state import GraphState
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated, Optional


# --- LangGraph + LangChain integration ---
settings = get_settings()
llm = init_chat_model(
    settings.LLM_MODEL,
    api_key=settings.openai_api_key,
    temperature=settings.LLM_TEMPERATURE,
    top_p=settings.LLM_TOP_P,
)


class State(GraphState):
    needs_medical_instructions: Optional[str] = None
    next: Optional[str] = None
    conversation_id: Optional[str] = None
    state_manager: Optional[object] = None


# Node: rag_desider
def rag_desider(state: State):
    # Get the last message from conversation history
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": "You are a classifier. Answer Yes if the user's message requires medical instructions (e.g., injury, medical kit, first aid, symptoms, emergency, etc.), and No if it is unrelated to medicine."},
        {"role": "user", "content": last_message["content"]},
    ]
    # Use a deterministic LLM for classification
    llm_zero_temp = init_chat_model(
        settings.LLM_MODEL,
        api_key=settings.openai_api_key,
        temperature=0,
        top_p=settings.LLM_TOP_P,
        max_tokens=10,
    )
    reply = llm_zero_temp.invoke(messages)
    answer = reply.content.strip().split()[0]  # Take first word (Yes/No)
    
    # Update state with classification
    state["needs_medical_instructions"] = answer
    return state


def router(state: State):
    rag_decision = state.get("needs_medical_instructions", "No")
    if rag_decision.lower() == "yes":
        state["next"] = "invoke_rag"
        return state

    state["next"] = "no_rag_chatter"
    return state


def invoke_rag(state: State):
    # Include conversation history for context
    system_message = {"role": "system", "content": INSTRUCTOR_PROMPT}
    messages = [system_message] + state["messages"][-settings.MAX_MESSAGES_FOR_MEMORY:]
    
    reply = llm.invoke(messages)
    
    state["state_manager"].add_message(state["conversation_id"], "assistant", reply.content)
    # Refresh state to get the updated messages with proper IDs
    updated_state = state["state_manager"].get_state(state["conversation_id"])
    state["messages"] = updated_state["messages"]
    return state


def no_rag_chatter(state: State):
    # Include conversation history for context
    system_message = {"role": "system", "content": CHATTER_PROMPT}
    messages = [system_message] + state["messages"][-settings.MAX_MESSAGES_FOR_MEMORY:]
    
    reply = llm.invoke(messages)
    
    state["state_manager"].add_message(state["conversation_id"], "assistant", reply.content)
    # Refresh state to get the updated messages with proper IDs
    updated_state = state["state_manager"].get_state(state["conversation_id"])
    state["messages"] = updated_state["messages"]
    return state


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("rag_desider", rag_desider)
graph_builder.add_node("router", router)
graph_builder.add_node("instructor", invoke_rag)
graph_builder.add_node("chatter", no_rag_chatter)

# Define the flow
graph_builder.add_edge(START, "rag_desider")
graph_builder.add_edge("rag_desider", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"invoke_rag": "instructor", "no_rag_chatter": "chatter"}
)
graph_builder.add_edge("instructor", END)
graph_builder.add_edge("chatter", END)

# Compile the graph
graph = graph_builder.compile()


async def generate(state: State, conversation_id: str, state_manager: object) -> dict:
    """
    Generate a response using the LangGraph pipeline.
    
    Args:
        user_query: The user's input message
        state: The current conversation state from LangGraphStateManager
        conversation_id: Optional ID of the conversation
        state_manager: Optional state manager instance for persisting changes
        
    Returns:
        dict: Contains the answer and updated state information
    """
    # Pass through state manager and conversation id
    state["conversation_id"] = conversation_id
    state["state_manager"] = state_manager
    
    # Run the graph with the current state - user message already added by websocket handler
    result = graph.invoke(state)
    
    if result.get("messages") and len(result["messages"]) > 0:
        # The state now contains the full conversation history
        last_message = result["messages"][-1]
        return {
            "needs_medical_instructions": result.get("needs_medical_instructions", ""),
            "answer": last_message["content"],
            "all_there_is_in_last_message": last_message
        }
    return {"answer": "Error when generating the reply", "needs_medical_instructions": None}