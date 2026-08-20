from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Replace with your pulled embedding model (e.g., "nomic-embed-text", "all-minilm", "mxbai-embed-large")
EMBEDDING_MODEL = "nomic-embed-text:latest"

# 1. Initialize the Embeddings abstraction
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://localhost:11434",
)


def exercise_1_embed_query():
    print("=== Exercise 1: embed_query() ===")
    query = "How does FastAPI dependency injection work?"

    query_vector = embeddings.embed_query(query)

    print(f"Type:              {type(query_vector)}")
    print(f"Dimension:         {len(query_vector)}")
    print(f"First 5 elements:  {query_vector[:5]}\n")
    return query_vector


def exercise_2_embed_documents():
    print("=== Exercise 2: embed_documents() Batch Processing ===")
    texts = [
        "FastAPI is a modern Python web framework for building APIs.",
        "Qdrant is a production-grade vector database with metadata filtering.",
        "LangChain standardizes model interfaces, tools, and retrieval pipelines.",
        "PostgreSQL uses MVCC to manage concurrent database transactions.",
    ]

    vectors = embeddings.embed_documents(texts)

    print(f"Number of input texts: {len(texts)}")
    print(f"Number of vectors:     {len(vectors)}")

    # Verify dimension uniformity
    dimensions = {len(v) for v in vectors}
    print(f"Unique dimensions:     {dimensions}\n")
    return texts, vectors


def exercise_3_ingestion_pipeline():
    print("=== Exercise 3: Document -> Splitter -> Embeddings Pipeline ===")

    raw_doc = Document(
        page_content=(
            "FastAPI relies on Starlette and Pydantic for high-performance API routing and schema validation. "
            "Qdrant stores dense and sparse embeddings alongside arbitrary JSON payload filters. "
            "LangChain provides the glue code to link split documents directly into vector stores."
        ),
        metadata={
            "source": "tech_stack.md",
            "author": "engineering",
            "tenant_id": "tenant_101",
        },
    )

    # 1. Split Document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=20
    )
    chunks: list[Document] = splitter.split_documents([raw_doc])

    # 2. Extract textual payloads for the embedding model
    chunk_texts = [chunk.page_content for chunk in chunks]

    # 3. Generate dense vectors
    chunk_vectors = embeddings.embed_documents(chunk_texts)

    print(f"Generated {len(chunks)} chunks from 1 parent document.")
    for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors), 1):
        print(f"--- Chunk {i} ---")
        print(f"Text:      {chunk.page_content}")
        print(f"Metadata:  {chunk.metadata}")
        print(f"Dim:       {len(vector)} | Vector Sample: {vector[:3]}...")
    print()



if __name__ == "__main__":
    exercise_1_embed_query()
    exercise_2_embed_documents()
    exercise_3_ingestion_pipeline()
