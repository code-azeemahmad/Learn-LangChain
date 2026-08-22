from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field

# Configuration
MODEL_NAME = "gemma4:26b"  # Ensure your local model supports tool calling (e.g., llama3, qwen2.5, mistral)
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_tools"

# =====================================================================
# 1. Setup Retrieval Backend for Knowledge Base Tool
# =====================================================================
def setup_vector_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")

    # Sample knowledge corpus
    docs = [
        Document(
            page_content="FastAPI dependency injection allows sharing database sessions and security dependencies across route handlers.",
            metadata={"source": "fastapi_docs.md"},
        ),
        Document(
            page_content="Enterprise plans include 24/7 dedicated support, custom SLAs, and multi-region failover clusters.",
            metadata={"source": "pricing_faq.md"},
        ),
    ]

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    return vector_store.as_retriever(search_kwargs={"k": 2})

retriever = setup_vector_retriever()


# =====================================================================
# 2. Tool Definitions with Explicit Schemas & Docstrings
# =====================================================================

# Tool 1: Deterministic Math Computation
@tool
def multiply(a: int, b: int,) -> int:
    """Multiply two integers together. Use this tool for exact arithmetic calculations."""
    return a * b


# Tool 2: Knowledge Base RAG Adapter
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for technical architecture and pricing documentation.
    Use this when answering company-specific questions that require verified source context.
    """
    docs: list[Document] = retriever.invoke(query)
    if not docs:
        return "No relevant internal documents found."
    
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# Tool 3: Simulated Database Lookup with Input Validation
class UserPlanInput(BaseModel):
    user_id: str = Field(
        description="The unique alphanumeric identifier of the user (e.g., 'usr_101', 'usr_102')."
    )

USER_DATABASE = {
    "usr_101": {"name": "Alice", "plan": "Enterprise", "active": True},
    "usr_102": {"name": "Bob", "plan": "Free Tier", "active": True},
}

@tool(args_schema=UserPlanInput)
def get_user_plan(user_id: str) -> str:
    """Look up a user's subscription tier and account status by their user ID."""
    user = USER_DATABASE.get(user_id)
    if not user:
        return f"User '{user_id}' not found in database."
    return f"User {user['name']} is on the {user['plan']} plan (Active: {user['active']})."


# =====================================================================
# 3. Exercises & Test Suite
# =====================================================================
def exercise_1_tool_inspection_and_direct_invocation():
    print("=== Exercise 1: Tool Schema Inspection & Direct Invocation ===")
    
    tools = [multiply, search_knowledge_base, get_user_plan]

    for t in tools:
        print(f"Tool Name:        {t.name}")
        print(f"Description:      {t.description.strip()}")
        print(f"Args Schema:      {t.args_schema.model_json_schema()['properties']}")
        print("-" * 50)

    # Direct synchronous execution without an LLM
    print("\n--- Direct Invocation (No LLM) ---")
    math_result = multiply.invoke({"a": 6, "b": 7})
    print(f"multiply.invoke({{'a': 6, 'b': 7}}) -> {math_result} (Type: {type(math_result)})")

    rag_result = search_knowledge_base.invoke({"query": "FastAPI dependency injection"})
    print(f"\nsearch_knowledge_base.invoke(...) ->\n{rag_result}")

    db_result = get_user_plan.invoke({"user_id": "usr_101"})
    print(f"\nget_user_plan.invoke({{'user_id': 'usr_101'}}) -> {db_result}\n")


def exercise_2_model_tool_binding():
    print("=== Exercise 2: Binding Tools to Chat Model (`bind_tools`) ===")
    
    base_model = ChatOllama(model=MODEL_NAME, temperature=0)

    # Bind tools to produce a tool-aware chat model
    model_with_tools = base_model.bind_tools(
        [multiply, search_knowledge_base, get_user_plan]
    )

    queries = [
        "What is 27 multiplied by 14?",
        "What features are included in the Enterprise plan according to internal docs?",
        "Can you check the subscription tier for user usr_102?",
        "What is the capital of France?",  # General knowledge query: should NOT trigger a tool call
    ]

    for q in queries:
        print(f"\nPrompt: '{q}'")
        response = model_with_tools.invoke(q)
        
        # Check if the model decided to request a tool call
        if response.tool_calls:
            print("  [TOOL CALL REQUESTED]")
            for tool_call in response.tool_calls:
                print(f"    - Function:  {tool_call['name']}")
                print(f"    - Arguments: {tool_call['args']}")
                print(f"    - ID:        {tool_call.get('id', 'N/A')}")
                print(response)
        else:
            print("  [DIRECT ANSWER (No Tool Called)]")
            print(f"    - Content: {response.content}")


if __name__ == "__main__":
    exercise_1_tool_inspection_and_direct_invocation()
    exercise_2_model_tool_binding()


"""
The Reasoning Behind Empty Content:
The model leaves content empty in Turn 1 because it cannot answer the user yet:
    For "What is 27 multiplied by 14?", the LLM recognized that computing the exact arithmetic is risky and delegated the computation to your multiply Python function.
    Because the Python function has not executed yet, the model does not know the result ($378$).
    Therefore, it has no text to place inside content. Its output is an execution request sent to your backend.
"""