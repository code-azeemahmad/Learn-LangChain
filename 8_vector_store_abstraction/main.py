from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# 1. Initialize Embeddings and direct Qdrant Client
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://localhost:11434",
)

client = QdrantClient(url=QDRANT_URL)


def setup_documents() -> list[Document]:
    return [
        Document(
            page_content="FastAPI is a modern, high-performance web framework for building APIs with Python and Pydantic.",
            metadata={"source": "fastapi.md", "topic": "backend", "tenant_id": "tenant_101"},
        ),
        Document(
            page_content="Qdrant is a vector similarity search engine that supports dense, sparse, and payload metadata filtering.",
            metadata={"source": "qdrant.md", "topic": "vector-db", "tenant_id": "tenant_101"},
        ),
        Document(
            page_content="LangChain standardizes orchestration interfaces across models, prompts, retrievers, and tool execution.",
            metadata={"source": "langchain.md", "topic": "ai-framework", "tenant_id": "tenant_101"},
        ),
        Document(
            page_content="PostgreSQL utilizes Multiversion Concurrency Control (MVCC) and WAL logging for durable ACID transactions.",
            metadata={"source": "postgres.md", "topic": "database", "tenant_id": "tenant_102"},
        ),
        Document(
            page_content="JSON Web Tokens (JWT) encode signed claims used for stateless authentication and RBAC authorization guards.",
            metadata={"source": "auth.md", "topic": "security", "tenant_id": "tenant_102"},
        ),
    ]


def exercise_1_index_documents():
    print("=== Exercise 1: Ingestion & Vector Store Initialization ===")
    
    docs = setup_documents()
    
    # from_documents initializes the collection (if not present) and upserts vectors + payloads
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,  # Clean slate for learning iterations
    )
    print(f"Successfully indexed {len(docs)} documents into collection '{COLLECTION_NAME}'.\n")
    return vector_store


def exercise_2_similarity_search(vector_store: QdrantVectorStore):
    print("=== Exercise 2: Similarity Search (similarity_search) ===")
    query = "How do I build a Python API?"
    
    # Direct vector search returning Document objects
    results: list[Document] = vector_store.similarity_search(query, k=2)
    
    print(f"Query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] Content: {doc.page_content}")
        print(f"    Metadata: {doc.metadata}")
    print()


def exercise_3_search_with_scores(vector_store: QdrantVectorStore):
    print("=== Exercise 3: Similarity Search With Scores ===")
    query = "How do vector databases work?"
    
    # Returns tuples of (Document, similarity_score)
    scored_results = vector_store.similarity_search_with_score(query, k=2)
    
    print(f"Query: '{query}'")
    for doc, score in scored_results:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content}")
        print("-" * 30)
    print()


def exercise_4_retriever_abstraction(vector_store: QdrantVectorStore):
    print("=== Exercise 4: Vector Store as a Retriever (Runnable Interface) ===")
    
    # Convert vector store to generic retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )
    
    query = "How do I authenticate users?"
    
    # .invoke() is the standardized Runnable entry point
    docs: list[Document] = retriever.invoke(query)
    
    print(f"Retriever Query (.invoke): '{query}'")
    for doc in docs:
        print(f"  - [{doc.metadata.get('topic')}]: {doc.page_content}")
    print()


def exercise_5_metadata_filtered_search(vector_store: QdrantVectorStore):
    print("=== Exercise 5: Tenant-Isolated Metadata Filter ===")
    
    # Qdrant native filter payload condition
    tenant_filter = rest_models.Filter(
        must=[
            rest_models.FieldCondition(
                key="metadata.tenant_id",
                match=rest_models.MatchValue(value="tenant_102"),
            )
        ]
    )
    
    query = "Explain databases and storage"
    filtered_results = vector_store.similarity_search(
        query,
        k=3,
        filter=tenant_filter,
    )
    
    print(f"Query under 'tenant_102' constraint: '{query}'")
    for doc in filtered_results:
        print(f"  - Tenant: {doc.metadata['tenant_id']} | Content: {doc.page_content}")
    print()


def test_dimension_mismatch_failure():
    print("=== Test: Dimension Mismatch Error Handling ===")
    
    # Pretend to use an embedding model with an incompatible dimension
    # (e.g., using a collection configured for 768 dims with a different provider/config)
    mismatched_embeddings = OllamaEmbeddings(
        model="llama3",  # Passing a chat model as an embedding model creates invalid vectors or dimensions
        base_url="http://localhost:11434",
    )
    
    try:
        bad_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=mismatched_embeddings,
        )
        bad_store.similarity_search("Test query")
    except Exception as e:  # noqa: BLE001
        print(f"Caught expected vector operation error: {type(e).__name__} -> {e}")


if __name__ == "__main__":
    # Ensure Qdrant is running on http://localhost:6333
    v_store = exercise_1_index_documents()
    exercise_2_similarity_search(v_store)
    exercise_3_search_with_scores(v_store)
    exercise_4_retriever_abstraction(v_store)
    exercise_5_metadata_filtered_search(v_store)
    test_dimension_mismatch_failure()


    """
    retriever.invoke(): 
        Input: str(query) 
        Output: list[Document]	
        Working: Embeds query, calls vector DB similarity search, returns docs
    """