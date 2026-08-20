import uuid
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def exercise_1_manual_documents():
    print("=== Exercise 1: Explicit Document Creation ===")
    
    # 1. Manually instantiate documents (e.g., from a PostgreSQL query or API)
    doc = Document(
        page_content="PostgreSQL handles concurrent transactions using Multiversion Concurrency Control (MVCC).",
        metadata={
            "user_id": "usr_9876",
            "source": "db_internals_guide.md",
            "category": "database",
            "version": 1.0,
        },
        id=str(uuid.uuid4()),
    )
    
    print(f"Document ID:   {doc.id}")
    print(f"Metadata:      {doc.metadata}")
    print(f"Page Content:  {doc.page_content}\n")


def exercise_2_text_loader():
    print("=== Exercise 2: File Ingestion via TextLoader ===")
    
    sample_file = Path("sample.txt")
    sample_file.write_text(
        "FastAPI uses Starlette for web routing and Pydantic for data validation.\n"
        "Dependency injection in FastAPI allows reusable database sessions and authentication guards.",
        encoding="utf-8",
    )

    try:
        # Load using standard TextLoader
        loader = TextLoader(str(sample_file), encoding="utf-8")
        
        # In-memory full load
        docs = loader.load()
        print(f"Loaded {len(docs)} document(s)")
        print(f"Type:     {type(docs[0])}")
        print(f"Metadata: {docs[0].metadata}")
        print(f"Content:\n{docs[0].page_content}\n")
        
        # Generator / lazy load pattern (production memory-safe)
        print("--- Lazy Loading Stream ---")
        for lazy_doc in loader.lazy_load():
            print(f"Lazy Loaded Doc Source: {lazy_doc.metadata['source']}")
    finally:
        if sample_file.exists():
            sample_file.unlink()


def exercise_3_metadata_enrichment():
    print("\n=== Exercise 3: Metadata Enrichment for Multi-Tenant RAG ===")
    
    raw_texts = [
        "User 101 secret API keys and billing info.",
        "User 102 deployment logs and cluster configuration.",
    ]
    
    user_ids = ["usr_101", "usr_102"]
    
    # Simulating document processing pipeline with strict tenant isolation
    processed_documents: list[Document] = []
    for text, uid in zip(raw_texts, user_ids):
        doc = Document(
            page_content=text,
            metadata={
                "tenant_id": uid,
                "access_level": "admin",
                "indexed_at": "2026-08-20",
            },
            id=str(uuid.uuid4()),
        )
        processed_documents.append(doc)

    for d in processed_documents:
        print(f"ID: {d.id} | Tenant: {d.metadata['tenant_id']} | Content: {d.page_content}")


if __name__ == "__main__":
    exercise_1_manual_documents()
    exercise_2_text_loader()
    exercise_3_metadata_enrichment()