# lesson20/app/ai.py
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient  # noqa: F401


def init_vector_store(
    qdrant_url: str,
    collection_name: str,
    embedding_model: str,
) -> QdrantVectorStore:
    """Provisions Qdrant collection with seed documents."""
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url="http://localhost:11434",
    )
    
    docs = [
        Document(
            page_content="FastAPI dependency injection uses 'Depends()' for database connections and security context.",
            metadata={"source": "fastapi_di.md", "topic": "framework"},
        ),
        Document(
            page_content="PostgreSQL utilizes Multiversion Concurrency Control (MVCC) and WAL for durable transactions.",
            metadata={"source": "postgres_internals.md", "topic": "database"},
        ),
        Document(
            page_content="Qdrant supports HNSW vector indexing with exact scalar payload filtering for tenant isolation.",
            metadata={"source": "qdrant_hnsw.md", "topic": "vector-db"},
        ),
    ]

    return QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        collection_name=collection_name,
        force_recreate=True,
    )


def create_ai_agent(
    model_name: str,
    vector_store: QdrantVectorStore,
):
    """Builds and compiles the stateful LangGraph agent graph."""
    
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search internal architecture, framework, and database documentation.
        Use this when questions require specific technical or system details.
        """
        docs = vector_store.similarity_search(query, k=2)
        if not docs:
            return "No relevant internal documentation found."
        return "\n\n".join(
            f"[{d.metadata.get('source', 'doc')}]: {d.page_content}"
            for d in docs
        )

    model = ChatOllama(model=model_name, temperature=0)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        checkpointer=checkpointer,
        system_prompt=(
            "You are a technical assistant. "
            "Use the 'search_knowledge_base' tool when technical or architecture details are required. "
            "Be precise, professional, and concise."
        ),
    )
    return agent