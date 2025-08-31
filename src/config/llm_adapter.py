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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model




MESSAGE_TEMPLATE = "Question: {query}\nContext: {context}"


# --- LangGraph + LangChain integration ---
settings = get_settings()
llm = init_chat_model(
    settings.LLM_MODEL,
    api_key=settings.openai_api_key,
    temperature=settings.LLM_TEMPERATURE,
    top_p=settings.LLM_TOP_P,
)


from typing import TypedDict, Annotated


class State(TypedDict):
    messages: Annotated[list, add_messages]
    needs_medical_instructions: str | None
    rag_decision: str | None


# Node: rag_desider
def rag_desider(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": "You are a classifier. Answer Yes if the user's message requires medical instructions (e.g., injury, medical kit, first aid, symptoms, emergency, etc.), and No if it is unrelated to medicine."},
        {"role": "user", "content": last_message.content},
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
    return {"needs_medical_instructions": answer, "messages": state["messages"]}

def router(state: State):
    rag_decision = state.get("needs_medical_instructions", "No")
    if rag_decision.lower() == "yes":
        return {"next": "invoke_rag"}

    return {"next": "no_rag_chatter"}


def invoke_rag(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": INSTRUCTOR_PROMPT},
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply.content}]}


def no_rag_chatter(state: State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": CHATTER_PROMPT},
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)
graph_builder.add_node("rag_desider", rag_desider)
graph_builder.add_node("router", router)
graph_builder.add_node("instructor", invoke_rag)
graph_builder.add_node("chatter", no_rag_chatter)
graph_builder.add_edge(START, "rag_desider")
graph_builder.add_edge("rag_desider", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"invoke_rag": "instructor", "no_rag_chatter": "chatter"}
)
graph_builder.add_edge("instructor", END)
graph_builder.add_edge("chatter", END)
graph = graph_builder.compile()


def generate(query: str, context: str) -> str:
    user_message = MESSAGE_TEMPLATE.format(query=query, context=context)
    state = {"messages": [{"role": "user", "content": user_message}]}
    result = graph.invoke(state)
    if result.get("messages") and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        return {
        "needs_medical_instructions": result.get("needs_medical_instructions", ""),
        "answer": last_message.content if result.get("messages") else "",
        "all_there_is_in_last_message": last_message
    }
    return "Error when generating the reply"


# Interactive chat loop
if __name__ == "__main__":
    print("HealthMate chat. Type 'q' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() == "q":
            print("Bye!")
            break
        # For now, context is empty or can be extended to use KB
        context = ""
        result = generate(user_query, context)
        print(f"Medical instructions needed: {result['needs_medical_instructions']}")
        print(f"Assistant: {result['answer']}")
