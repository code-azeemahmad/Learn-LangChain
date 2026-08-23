from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

# Configuration
MODEL_NAME = "gemma4:26b"


# 1. Define the Tool Capability
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use this tool for exact arithmetic."""
    return a * b


def build_stateful_agent():
    model = ChatOllama(model=MODEL_NAME, temperature=0)

    # 2. Instantiate In-Memory Checkpoint Saver
    # Persists graph state snapshots in RAM keyed by thread_id
    checkpointer = InMemorySaver()

    # 3. Create the Agent with Checkpointing Enabled
    agent = create_agent(
        model=model,
        tools=[multiply],
        checkpointer=checkpointer,
        system_prompt=(
            "You are a helpful and precise assistant. "
            "Always use the multiply tool for arithmetic operations. "
            "Refer to previous conversation turns when answering follow-up questions."
        ),
    )
    return agent


# =====================================================================
# State Inspection Helper
# =====================================================================
def inspect_thread_state(result: dict, thread_label: str):
    print(f"\n--- Current Graph State for [{thread_label}] ---")
    messages = result.get("messages", [])
    print(f"Total Messages in State: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        tool_call_info = (
            f" | Tool Calls: {msg.tool_calls}"
            if hasattr(msg, "tool_calls") and msg.tool_calls
            else ""
        )
        print(f"  [{i}] {msg_type}: {msg.content}{tool_call_info}")
    print("-" * 60)


# =====================================================================
# Test Suite: Multi-Turn Memory & Thread Isolation
# =====================================================================
def main():
    agent = build_stateful_agent()

    # -------------------------------------------------------------
    # Test Suite 1: Thread-A Multi-Turn State Accumulation
    # -------------------------------------------------------------
    config_a = {"configurable": {"thread_id": "thread-A"}}

    print("=== [Thread-A] Turn 1: Initial Calculation ===")
    q1 = "What is 8 multiplied by 9?"
    print(f"User: {q1}")
    result_a1 = agent.invoke({"messages": [HumanMessage(content=q1)]}, config=config_a)
    print(f"AI:   {result_a1['messages'][-1].content}")

    print("\n=== [Thread-A] Turn 2: Contextual Follow-up ===")
    q2 = "Now multiply that previous result by 2."
    print(f"User: {q2}")
    # We only send the NEW message. LangGraph loads prior turns via thread_id.
    result_a2 = agent.invoke({"messages": [HumanMessage(content=q2)]}, config=config_a)
    print(f"AI:   {result_a2['messages'][-1].content}")

    inspect_thread_state(result_a2, "thread-A")

    # -------------------------------------------------------------
    # Test Suite 2: Thread-B State Isolation Verification
    # -------------------------------------------------------------
    config_b = {"configurable": {"thread_id": "thread-B"}}

    print("\n=== [Thread-B] Turn 1: Cross-Thread Isolation Test ===")
    q_b1 = "What was the previous calculation result we just computed?"
    print(f"User: {q_b1}")
    result_b1 = agent.invoke(
        {"messages": [HumanMessage(content=q_b1)]}, config=config_b
    )
    print(f"AI:   {result_b1['messages'][-1].content}")
    # Expected: The model does not know about Thread-A's 72 or 144.

    print("\n=== [Thread-B] Turn 2: Independent Work in Thread-B ===")
    q_b2 = "What is 10 multiplied by 10?"
    print(f"User: {q_b2}")
    result_b2 = agent.invoke(
        {"messages": [HumanMessage(content=q_b2)]}, config=config_b
    )
    print(f"AI:   {result_b2['messages'][-1].content}")

    inspect_thread_state(result_b2, "thread-B")


if __name__ == "__main__":
    main()


"""
[ Frontend / React Client ]
       │  (Sends message + conversation_id)
       ▼
[ FastAPI Backend / Service Layer ]
       │  (Maps conversation_id -> thread_id)
       ▼
[ LangGraph Agent Runtime ] ◄───► [ Checkpointer (Postgres / InMemory) ]
   (Maintains full dialogue, tool calls, and execution state across turns)
       │
       ├─► Direct Answer (if general query)
       │
       └─► Tool Call: search_knowledge_base(query)
                 │
                 ▼
          [ RAG Retrieval Sub-Pipeline ]
          1. Inspect recent dialogue state
          2. Query Rewriter: "Can it scale?" ──► "PostgreSQL connection pooling scaling"
          3. Dense / Sparse Search in Qdrant
          4. Returns relevant chunks
                 │
                 ▼
          [ ToolMessage returned to Agent ]
                 │
                 ▼
          Agent synthesizes final grounded answer
"""