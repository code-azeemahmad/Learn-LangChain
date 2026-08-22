from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:26b"

# 1. Define the Tool Capability
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use this tool for exact arithmetic multiplications."""
    return a * b


def build_minimal_agent():
    # 2. Initialize Model
    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    # 3. Create the Agent State Graph
    # Returns a CompiledStateGraph (LangGraph runtime underneath)
    agent = create_agent(
        model=model,
        tools=[multiply],
        system_prompt=(
            "You are a helpful and precise assistant. "
            "Always use the multiply tool when asked to perform multiplication."
        ),
    )
    return agent


# =====================================================================
# Test Suite & State Inspection
# =====================================================================
def exercise_1_tool_execution_loop(agent):
    print("=== Test 1: Tool Execution Cycle (Inspection of State Messages) ===")
    query = "What is 17 multiplied by 23?"
    print(f"User Query: '{query}'\n")

    # Agent takes a dict with the conversation messages
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    print("--- Message Sequence in Final Agent State ---")
    for i, message in enumerate(result["messages"], 1):
        msg_type = type(message).__name__
        print(f"[{i}] {msg_type}:")
        
        # Display text content if present
        if message.content:
            print(f"    Content: {message.content}")
            
        # Display tool call instructions emitted by the model
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"    Tool Calls Requested: {message.tool_calls}")
            
        # Display tool call ID link for ToolMessages
        if hasattr(message, "tool_call_id") and message.tool_call_id:
            print(f"    Linked Tool Call ID:  {message.tool_call_id}")
            
        print("-" * 50)


def exercise_2_direct_reasoning_no_tool(agent):
    print("\n=== Test 2: General Knowledge (No Tool Invocation) ===")
    query = "What is Python in one sentence?"
    print(f"User Query: '{query}'\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    print(f"Total Messages in State: {len(result['messages'])}")
    final_message = result["messages"][-1]
    print(f"Final AIMessage Content:\n{final_message.content}\n")
    
    # Verify no tool messages were injected into state
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    print(f"Tool Messages Injected: {len(tool_messages)} (Expected: 0)")


def exercise_3_stream_graph_updates(agent):
    print("\n=== Test 3: Streaming Graph State Updates (`stream_mode='updates'`) ===")
    query = "What is 12 multiplied by 8?"
    print(f"Streaming Steps for: '{query}'\n")

    # Streams node-level execution transitions as they occur in LangGraph
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates",
    ):
        for node_name, node_update in chunk.items():
            print(f"--- Node Executed: [{node_name}] ---")
            for msg in node_update.get("messages", []):
                print(f"  {type(msg).__name__} -> Content: '{msg.content}' | Tool Calls: {getattr(msg, 'tool_calls', [])}")
        print()


if __name__ == "__main__":
    agent_graph = build_minimal_agent()
    
    exercise_1_tool_execution_loop(agent_graph)
    exercise_2_direct_reasoning_no_tool(agent_graph)
    exercise_3_stream_graph_updates(agent_graph)