import asyncio

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# Configuration
MODEL_NAME = "gemma4:26b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_lcel"

# Shared Model & Parser Instances
model = ChatOllama(model=MODEL_NAME, temperature=0)
parser = StrOutputParser()


# =====================================================================
# Helper: Document Store & Retriever Setup for Part C
# =====================================================================
def setup_qdrant_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    docs = [
        Document(
            page_content="FastAPI dependency injection uses 'Depends()' to declare reusable dependencies like DB sessions or auth guards.",
            metadata={"source": "fastapi_di.md"},
        ),
        Document(
            page_content="Qdrant provides vector indexing with HNSW and exact payload metadata filtering at search time.",
            metadata={"source": "qdrant_indexing.md"},
        ),
        Document(
            page_content="LCEL (LangChain Expression Language) allows deterministic composition of runnables using the pipe operator.",
            metadata={"source": "lcel_guide.md"},
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


# =====================================================================
# Exercise 1 (Part A): Sequential Composition & All Execution Modes
# =====================================================================
async def exercise_1_sequential_chain():
    print("=== Exercise 1: Sequential Composition & Execution Modes ===")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise Python tutor. Keep answers under 2 sentences."),
            ("human", "Explain {topic} in simple terms."),
        ]
    )

    # Declarative composition: dict -> ChatPromptTemplate -> ChatOllama -> StrOutputParser -> str
    chain = prompt | model | parser

    # 1. Synchronous invoke
    sync_result = chain.invoke({"topic": "decorators"})
    print(f"1. .invoke() Output:\n{sync_result}\n")

    # 2. Asynchronous ainvoke
    async_result = await chain.ainvoke({"topic": "generators"})
    print(f"2. .ainvoke() Output:\n{async_result}\n")

    # 3. Synchronous stream
    print("3. .stream() Output: ", end="", flush=True)
    for chunk in chain.stream({"topic": "list comprehensions"}):
        print(chunk, end="", flush=True)
    print("\n")

    # 4. Asynchronous astream
    print("4. .astream() Output: ", end="", flush=True)
    async for chunk in chain.astream({"topic": "asyncio event loops"}):
        print(chunk, end="", flush=True)
    print("\n")


# =====================================================================
# Exercise 2 (Part B): Concurrent Branching via RunnableParallel
# =====================================================================
def exercise_2_parallel_branches():
    print("=== Exercise 2: Parallel Branch Execution (RunnableParallel) ===")

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in 5 words or less:\n{text}"
    )
    classification_prompt = ChatPromptTemplate.from_template(
        "Classify the following text into exactly ONE category [Technical, Business, General]:\n{text}"
    )

    summary_chain = summary_prompt | model | parser
    classification_chain = classification_prompt | model | parser

    # Parallel mapping: dispatches the same input payload to both branches concurrently
    parallel_workflow = RunnableParallel(
        summary=summary_chain,
        classification=classification_chain,
    )

    input_data = {
        "text": "PostgreSQL implements multi-version concurrency control (MVCC) to ensure ACID compliance during concurrent writes."
    }

    result = parallel_workflow.invoke(input_data)
    print(f"Input: {input_data['text']}")
    print(f"Parallel Result Type: {type(result)}")
    print(f"  - Summary:        {result['summary'].strip()}")
    print(f"  - Classification: {result['classification'].strip()}\n")


# =====================================================================
# Exercise 3 (Part C): Complete Deterministic RAG Chain
# =====================================================================
def exercise_3_rag_pipeline(retriever):
    print("=== Exercise 3: Deterministic RAG Composition ===")

    # Custom context formatter wrapped into a Runnable
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"Source [{doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
            for doc in docs
        )

    format_docs_runnable = RunnableLambda(format_docs)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user question strictly using the context below. Keep it under 2 sentences.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # RAG LCEL Data-Flow Architecture:
    # 1. Input string passes to retriever (via format_docs) and RunnablePassthrough concurrently.
    # 2. Results merged into {"context": str, "question": str}.
    # 3. Passed to rag_prompt -> model -> parser.
    rag_chain = (
        {
            "context": retriever | format_docs_runnable,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | model
        | parser
    )

    query = "How does FastAPI handle dependency injection?"
    print(f"RAG Query: '{query}'")
    answer = rag_chain.invoke(query)
    print(f"RAG Answer:\n{answer}\n")

    # Inspect the execution graph
    print("--- Execution Graph (ASCII) ---")
    try:
        print(rag_chain.get_graph().draw_ascii())
    except Exception:  # noqa: BLE001
        print("Note: Optional dependencies for graph ASCII drawing not installed (pygraphviz/grandalf).")


async def main():
    await exercise_1_sequential_chain()
    exercise_2_parallel_branches()
    
    retriever = setup_qdrant_retriever()
    exercise_3_rag_pipeline(retriever)


if __name__ == "__main__":
    asyncio.run(main())

"""
answer = rag_chain.invoke(query)
rag_chain = (
    {
        "context": retriever | format_docs_runnable,
        "question": RunnablePassthrough(),
    }
)

Think of this as a pipeline with two stages:

Stage 1 — a dictionary that builds {"context": ..., "question": ...} from the raw input query.
Stage 2 — that dictionary flows into rag_prompt → model → parser in sequence.

Step-by-step, when you call rag_chain.invoke(query)

query = "How does FastAPI handle dependency injection?" is the single input string.

1. The dictionary stage runs both branches on the same input

In LCEL, when you write a plain Python dict inside a chain, LangChain treats it as a parallel runnable (RunnableParallel). The same input (query) is fed to every value in the dict at once:

"context" branch: retriever | format_docs_runnable
retriever.invoke(query) — the query string goes into the retriever, which returns list[Document] (the relevant docs from your vector store).
Those docs are piped into format_docs_runnable, which calls your format_docs() function, turning the list of Document objects into a single formatted string like:
     Source [fastapi.md]:
     FastAPI is a Python web framework...
     
     Source [qdrant.md]:
     Qdrant is a dedicated vector database...
"question" branch: RunnablePassthrough()
This just returns the input unchanged — so "question" becomes the original query string, untouched.

After this stage, the pipeline holds:

python
{
    "context": "Source [fastapi.md]:\nFastAPI is a Python web framework...",
    "question": "How does FastAPI handle dependency injection?"
}

"""
