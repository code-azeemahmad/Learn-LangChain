import asyncio

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL = "nomic-embed-text:latest"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://localhost:11434",
)

client = QdrantClient(url=QDRANT_URL)


def setup_sample_documents() -> list[Document]:
    return [
        Document(
            page_content="FastAPI is a Python web framework built for async REST APIs and OpenAPI docs.",
            metadata={"source": "fastapi.md", "topic": "backend"},
        ),
        Document(
            page_content="Qdrant is a dedicated vector database supporting HNSW indexing and payload filtering.",
            metadata={"source": "qdrant.md", "topic": "vector-db"},
        ),
        Document(
            page_content="PostgreSQL supports relational tables, JSONB columns, and ACID transactions.",
            metadata={"source": "postgres.md", "topic": "database"},
        ),
    ]


# =====================================================================
# Task B: Custom BaseRetriever Implementation (simplified)
# =====================================================================
class KeywordRetriever(BaseRetriever):
    """A simple keyword-matching retriever."""

    documents: list[Document]
    top_k: int = 3

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Called when you use retriever.invoke(query)."""
        query_words = set(query.lower().split())
        matched_docs = []

        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = query_words & doc_words
            if overlap:
                matched_docs.append((doc, len(overlap)))

        # Sort by number of matching words, most matches first
        matched_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in matched_docs[: self.top_k]]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Called when you use await retriever.ainvoke(query)."""
        return self._get_relevant_documents(query)


# =====================================================================
# Task A: Vector Store as Retriever
# =====================================================================
def exercise_qdrant_retriever():
    print("=== Task A: Qdrant VectorStore as Retriever ===")
    docs = setup_sample_documents()

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    retriever_k2 = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever_k1 = vector_store.as_retriever(search_kwargs={"k": 1})

    query = "How do I build Python APIs?"

    results_k2 = retriever_k2.invoke(query)
    results_k1 = retriever_k1.invoke(query)

    print(f"Query: '{query}'")
    print(f"k=2 returned {len(results_k2)} Document(s):")
    for d in results_k2:
        print(f"  - [{d.metadata['topic']}]: {d.page_content}")

    print(f"k=1 returned {len(results_k1)} Document(s):")
    for d in results_k1:
        print(f"  - [{d.metadata['topic']}]: {d.page_content}\n")

    assert len(results_k2) == 2, "k=2 retrieval failed"
    assert len(results_k1) == 1, "k=1 retrieval failed"
    assert isinstance(results_k2[0], Document), "Return type is not Document"


# =====================================================================
# Custom Retriever Execution
# =====================================================================
async def exercise_custom_keyword_retriever():
    print("=== Task B: Custom KeywordRetriever Execution & Testing ===")
    docs = setup_sample_documents()
    custom_retriever = KeywordRetriever(documents=docs, top_k=2)

    sync_results = custom_retriever.invoke("Python FastAPI framework")
    print(f"Sync invoke returned {len(sync_results)} doc(s):")
    for d in sync_results:
        print(f"  - {d.page_content} (metadata: {d.metadata})")

    empty_results = custom_retriever.invoke("Quantum physics thermodynamics")
    print(f"Unrelated query returned {len(empty_results)} doc(s) (Expected: 0)")
    assert len(empty_results) == 0, "Empty retrieval test failed."

    print("\n--- Testing Async Retrieval (.ainvoke) ---")
    async_results = await custom_retriever.ainvoke("relational tables JSONB")
    print(f"Async ainvoke returned {len(async_results)} doc(s):")
    for d in async_results:
        print(f"  - Topic: {d.metadata.get('topic')} | Content: {d.page_content}")
        assert "topic" in d.metadata, "Metadata was dropped during retrieval"
    print()


async def main():
    exercise_qdrant_retriever()
    await exercise_custom_keyword_retriever()
    print("All Retriever checks passed successfully.")


if __name__ == "__main__":
    asyncio.run(main())