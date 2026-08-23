import asyncio
import json
from typing import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

# Configuration
# Use a tool-capable local model (e.g., llama3, qwen2.5, mistral)
MODEL_NAME = "gemma4:26b"


# =====================================================================
# 1. Tool & Stateful Agent Setup
# =====================================================================
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use this tool for exact arithmetic."""
    return a * b


def build_streaming_agent():
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[multiply],
        checkpointer=checkpointer,
        system_prompt=(
            "You are a precise technical tutor. "
            "Always use the multiply tool for arithmetic operations. "
            "Be concise."
        ),
    )
    return agent


# =====================================================================
# Experiment 1: State Update Streaming (stream_mode="updates")
# =====================================================================
def exercise_1_stream_updates(agent):
    print("=== Experiment 1: Execution State Updates (stream_mode='updates') ===")
    query = "What is 14 multiplied by 19?"
    print(f"User Query: '{query}'\n")

    config = {"configurable": {"thread_id": "stream-thread-1"}}

    # Yields state diffs after each graph node completes
    for update in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_state in update.items():
            print(f"[Node Completed: '{node_name}']")
            for msg in node_state.get("messages", []):
                msg_type = type(msg).__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"  -> {msg_type}: Requesting tool calls: {msg.tool_calls}")
                elif hasattr(msg, "tool_call_id"):
                    print(f"  -> {msg_type} (Result): {msg.content}")
                else:
                    print(f"  -> {msg_type} (Output): {msg.content}")
        print("-" * 50)
    print()


# =====================================================================
# Experiment 2: Token / Message Chunk Streaming (stream_mode="messages")
# =====================================================================
def exercise_2_stream_messages(agent):
    print("=== Experiment 2: LLM Token Chunk Streaming (stream_mode='messages') ===")
    query = "Explain what an event loop is in one concise sentence."
    print(f"User Query: '{query}'\nStreaming Tokens: ", end="", flush=True)

    config = {"configurable": {"thread_id": "stream-thread-2"}}

    # Yields tuples of (message_chunk, metadata) directly from the model
    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n\n" + "-" * 50 + "\n")


# =====================================================================
# Experiment 3: Asynchronous Graph Streaming (agent.astream)
# =====================================================================
async def exercise_3_async_stream(agent):
    print("=== Experiment 3: Async Execution Streaming (.astream) ===")
    query = "What is 9 multiplied by 13?"
    print(f"Async Query: '{query}'\n")

    config = {"configurable": {"thread_id": "stream-thread-3"}}

    # Non-blocking async generator for FastAPI event loops
    async for event in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_state in event.items():
            print(f"Async Node Executed: [{node_name}] -> Delta items: {len(node_state.get('messages', []))}")
    print("\n" + "-" * 50 + "\n")


# =====================================================================
# Experiment 4: Production SSE Event Adapter Preview
# =====================================================================
async def sse_event_adapter(
    agent, query: str, thread_id: str
) -> AsyncGenerator[str, None]:
    """
    Normalizes raw LangGraph execution events into an application-level SSE contract.
    Decouples frontend clients from framework-specific internal payloads.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Emit session start event
    yield f"data: {json.dumps({'type': 'status', 'stage': 'agent_started'})}\n\n"

    async for chunk, metadata in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        # 1. Model is streaming intermediate reasoning / tool calls
        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                if tc.get("name"):
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tc['name']})}\n\n"

        # 2. Model is streaming final answer tokens
        elif isinstance(chunk, AIMessageChunk) and chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def exercise_4_sse_simulation(agent):
    print("=== Experiment 4: Production SSE Event Stream Simulation ===")
    query = "What is 25 multiplied by 4?"
    print(f"Simulating SSE Output for: '{query}'\n")

    async for sse_line in sse_event_adapter(agent, query, "stream-thread-4"):
        print(sse_line.strip())
    print()


async def main():
    agent = build_streaming_agent()

    exercise_1_stream_updates(agent)
    exercise_2_stream_messages(agent)
    await exercise_3_async_stream(agent)
    await exercise_4_sse_simulation(agent)


if __name__ == "__main__":
    asyncio.run(main())