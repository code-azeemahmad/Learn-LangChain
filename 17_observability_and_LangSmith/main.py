import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from qdrant_client import QdrantClient

load_dotenv()

# Configuration
MODEL_NAME = "gemma4:26b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_lesson18"

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL, 
    base_url="http://localhost:11434",
)
qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# =====================================================================
# 1. Custom Non-LangChain Logic Traced with @traceable
# =====================================================================
@traceable(name="calculate_relevance_score", tags=["custom_logic"])
def calculate_relevance_score(query: str, doc_count: int) -> dict:
    """Demonstrates how arbitrary Python functions appear inside the LangSmith trace tree."""
    return {
        "query_length": len(query),
        "docs_retrieved": doc_count,
        "is_sufficient": doc_count >= 2,
    }


# =====================================================================
# 2. Knowledge Base & Tool Setup
# =====================================================================
def setup_knowledge_base() -> QdrantVectorStore:
    docs = [
        Document(
            page_content="FastAPI dependency injection uses 'Depends()' to provide shared services like DB sessions and auth guards.",
            metadata={"source": "fastapi_di.md", "topic": "backend"},
        ),
        Document(
            page_content="PostgreSQL utilizes Multiversion Concurrency Control (MVCC) to handle high write-concurrency workloads.",
            metadata={"source": "postgres_mvcc.md", "topic": "database"},
        ),
        Document(
            page_content="Qdrant supports HNSW vector indexing with exact scalar payload filtering for tenant isolation.",
            metadata={"source": "qdrant_hnsw.md", "topic": "vector-db"},
        ),
    ]

    return QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use this tool for exact arithmetic."""
    return a * b


# =====================================================================
# Exercise 1: Tracing a RAG Pipeline with Tags and Metadata
# =====================================================================
def exercise_1_rag_trace(vector_store: QdrantVectorStore):
    print("=== Exercise 1: Tracing RAG Pipeline with Run Configuration ===")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs: list[Document]) -> str:
        # Run our custom @traceable diagnostic inside the execution tree
        calculate_relevance_score("FastAPI DI", len(docs))
        return "\n\n".join(f"[{doc.metadata['source']}]: {doc.page_content}" for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question strictly using the provided context. If unknown, say 'I do not have enough information.'\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    # Attach operational telemetry: tags, metadata, and custom run name
    config = {
        "run_name": "fastapi_dependency_injection_query",
        "tags": ["lesson-18", "rag_v1", "development"],
        "metadata": {
            "environment": "local_dev",
            "retrieval_k": 2,
            "user_id": "usr_dev_101",
        },
    }

    query = "How does FastAPI handle dependency injection?"
    print(f"Executing RAG Query: '{query}'")
    answer = rag_chain.invoke(query, config=config)
    print(f"RAG Response:\n{answer}\n")


# =====================================================================
# Exercise 2: Tracing a Tool-Calling Agent Execution Loop
# =====================================================================
def exercise_2_agent_trace():
    print("=== Exercise 2: Tracing Agent State Graph & Tool Loops ===")

    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[multiply],
        checkpointer=checkpointer,
        system_prompt="You are a precise assistant. Always use the multiply tool for arithmetic.",
    )

    config = {
        "run_name": "agent_multiplication_trace",
        "tags": ["agent", "langgraph", "tools"],
        "configurable": {"thread_id": "trace-thread-01"},
        "metadata": {"session_type": "math_evaluation"},
    }

    query = "What is 19 multiplied by 23?"
    print(f"Executing Agent Query: '{query}'")
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    print(f"Agent Final Answer: {result['messages'][-1].content}\n")


# =====================================================================
# Exercise 3: Diagnosing Upstream Retrieval Failures in Traces
# =====================================================================
def exercise_3_broken_retrieval_diagnostic(vector_store: QdrantVectorStore):
    print("=== Exercise 3: Diagnosing Deliberate Retrieval Failure ===")

    # Intentionally set k=1 to force an incomplete retrieval context
    constrained_retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    def format_docs(docs: list[Document]) -> str:
        calculate_relevance_score("Postgres + Qdrant", len(docs))
        return "\n\n".join(f"[{doc.metadata['source']}]: {doc.page_content}" for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user strictly using the context below. If missing, say 'I do not have enough information.'\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": constrained_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    config = {
        "run_name": "retrieval_failure_diagnostic",
        "tags": ["diagnostic", "intentional_failure"],
        "metadata": {"expected_failure": "k_too_low"},
    }

    # Query requires Postgres AND Qdrant information, but k=1 only retrieves one document
    query = "Explain how PostgreSQL handles MVCC and how Qdrant indexes vectors."
    print(f"Multi-Topic Query with k=1: '{query}'")
    answer = rag_chain.invoke(query, config=config)
    print(f"Response Under Constraint:\n{answer}\n")


def main():
    vstore = setup_knowledge_base()
    
    exercise_1_rag_trace(vstore)
    exercise_2_agent_trace()
    exercise_3_broken_retrieval_diagnostic(vstore)

    # Flush any background worker threads before the process terminates
    print("Flushing telemetry to LangSmith...")
    wait_for_all_tracers()
    print("Tracing complete. Inspect your dashboard at https://smith.langchain.com")


if __name__ == "__main__":
    main()