from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Multi-paragraph sample document
SAMPLE_TEXT = (
        "FastAPI is a modern Python web framework for building APIs. "
        "It uses Python type hints and provides automatic API documentation. "
        "FastAPI applications are commonly deployed behind an ASGI server. "
        "Dependency injection is an important part of FastAPI application design."
)

base_doc = Document(
    page_content=SAMPLE_TEXT,
    metadata={
        "source": "architecture_overview.txt",
        "document_id": "doc-001",
        "tenant_id": "tenant-001",
    },
)


def exercise_1_split_documents():
    print("=== Exercise 1: split_documents() with Metadata Preservation ===")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )

    chunks = splitter.split_documents([base_doc])
    print(f"Generated {len(chunks)} Document chunks.\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- CHUNK {i} ---")
        print(f"Content:\n{chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Character Count: {len(chunk.page_content)}")
        print("-" * 30)


def exercise_2_split_text_comparison():
    print("\n=== Exercise 2: split_text() vs split_documents() ===")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )

    # Low-level string splitting (returns raw strings, drops metadata)
    raw_text_chunks: list[str] = splitter.split_text(base_doc.page_content)
    print(f"split_text() returned {len(raw_text_chunks)} raw strings.")
    print(f"Type of item 0: {type(raw_text_chunks[0])}")
    print(f"Sample raw chunk:\n{raw_text_chunks[0][:80]}...\n")

    # Document splitting (returns Document objects, preserves metadata)
    doc_chunks: list[Document] = splitter.split_documents([base_doc])
    print(f"split_documents() returned {len(doc_chunks)} Document objects.")
    print(f"Type of item 0: {type(doc_chunks[0])}")
    print(f"Metadata preserved: {doc_chunks[0].metadata}")


def exercise_3_parameter_experiments():
    print("\n=== Exercise 3: Parameter Sweep Experiments ===")
    
    configs = [
        {"chunk_size": 100, "chunk_overlap": 20},
        {"chunk_size": 200, "chunk_overlap": 40},
        {"chunk_size": 500, "chunk_overlap": 50},
    ]

    for cfg in configs:
        exp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        exp_chunks = exp_splitter.split_documents([base_doc])
        print(
            f"Config [size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}] "
            f"--> Produced {len(exp_chunks)} chunks"
        )


if __name__ == "__main__":
    exercise_1_split_documents()
    exercise_2_split_text_comparison()
    exercise_3_parameter_experiments()