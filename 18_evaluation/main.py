# lesson19/main.py
import os
import time

from dataset import EVALUATION_DATASET
from dotenv import load_dotenv
from evaluator import (
    evaluate_correctness,
    evaluate_exact_boundary,
    evaluate_groundedness,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import Client, evaluate
from qdrant_client import QdrantClient

load_dotenv()

# Configuration
MODEL_NAME = "gemma4:26b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langchain_lesson19"
DATASET_NAME = "fastapi-rag-benchmark-v1"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
model = ChatOllama(model=MODEL_NAME, temperature=0)
qdrant_client = QdrantClient(url=QDRANT_URL)


# =====================================================================
# 1. Target RAG Pipeline Setup
# =====================================================================
def build_target_rag():
    docs = [
        Document(
            page_content="FastAPI is a modern, high-performance web framework for building APIs with Python and standard type hints.",
            metadata={"source": "fastapi_overview.md"},
        ),
        Document(
            page_content="Dependency injection in FastAPI uses the 'Depends' function to supply DB connections, security context, and configuration.",
            metadata={"source": "fastapi_di.md"},
        ),
        Document(
            page_content="FastAPI applications are executed using an ASGI server such as Uvicorn or Hypercorn.",
            metadata={"source": "fastapi_deployment.md"},
        ),
        Document(
            page_content="PostgreSQL utilizes Multiversion Concurrency Control (MVCC) and Write-Ahead Logging (WAL) for high concurrency ACID compliance.",
            metadata={"source": "postgres_internals.md"},
        ),
    ]

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(retrieved_docs: list[Document]) -> str:
        return "\n\n".join(f"[{d.metadata['source']}]: {d.page_content}" for d in retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question strictly using the provided context. If the answer is not present, "
                "state: 'I do not have enough information to answer this.'\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # Return raw components to allow inspection of both context and prediction
    return retriever, format_docs, prompt, model


# =====================================================================
# 2. Local Evaluation Execution Loop
# =====================================================================
def run_local_evaluation():
    print("=== Step 1: Running Local Multi-Dimensional Evaluation Loop ===\n")
    retriever, format_docs, prompt, rag_model = build_target_rag()

    total_correctness = 0.0
    grounded_count = 0
    latencies: list[float] = []

    for i, example in enumerate(EVALUATION_DATASET, 1):
        q = example["inputs"]["question"]
        ref = example["outputs"]["answer"]
        category = example["metadata"]["category"]

        t0 = time.perf_counter()
        
        # 1. Target Retrieval & Formatting
        retrieved_docs = retriever.invoke(q)
        formatted_context = format_docs(retrieved_docs)

        # 2. Target Generation
        messages = prompt.invoke({"context": formatted_context, "question": q})
        raw_response = rag_model.invoke(messages)
        prediction = raw_response.content.strip()
        
        latency = time.perf_counter() - t0
        latencies.append(latency)

        # 3. Evaluate with LLM-as-Judge & Heuristics
        correctness = evaluate_correctness(q, ref, prediction)
        groundedness = evaluate_groundedness(formatted_context, prediction)

        total_correctness += correctness.score
        if groundedness.is_grounded:
            grounded_count += 1

        print(f"--- Example [{i}/{len(EVALUATION_DATASET)}] ({category}) ---")
        print(f"Question:    {q}")
        print(f"Prediction:  {prediction}")
        print(f"Correctness: {correctness.score:.2f} (Reason: {correctness.reason})")
        print(f"Grounded:    {groundedness.is_grounded} (Reason: {groundedness.reason})")
        print(f"Latency:     {latency:.2f}s\n")

    n = len(EVALUATION_DATASET)
    avg_correctness = total_correctness / n
    groundedness_rate = (grounded_count / n) * 100
    avg_latency = sum(latencies) / n

    print("==================================================")
    print("           LOCAL EVALUATION SUMMARY REPORT        ")
    print("==================================================")
    print(f"Total Test Cases:    {n}")
    print(f"Mean Correctness:    {avg_correctness:.4f} / 1.0000")
    print(f"Groundedness Rate:   {groundedness_rate:.1f}%")
    print(f"Average Latency:     {avg_latency:.2f}s")
    print("==================================================\n")


# =====================================================================
# 3. LangSmith Platform Synchronization & Experiment Runner
# =====================================================================
def run_langsmith_experiment():
    print("=== Step 2: Uploading Dataset & Executing LangSmith Experiment ===")
    
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("Notice: LANGSMITH_API_KEY is not set. Skipping remote experiment execution.")
        return

    client = Client()

    # 1. Idempotently create or retrieve dataset in LangSmith
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Benchmark dataset for FastAPI and database RAG system",
        )
        for ex in EVALUATION_DATASET:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex["outputs"],
                metadata=ex["metadata"],
                dataset_id=dataset.id,
            )
        print(f"Created new LangSmith dataset: '{DATASET_NAME}' with {len(EVALUATION_DATASET)} examples.")
    else:
        print(f"Using existing LangSmith dataset: '{DATASET_NAME}'.")

    # 2. Define the Target Invocation Wrapper
    retriever, format_docs, prompt, rag_model = build_target_rag()
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | rag_model
        | StrOutputParser()
    )

    def target_application(inputs: dict) -> dict:
        result = rag_chain.invoke(inputs["question"])
        return {"answer": result}

    # 3. Define LangSmith Evaluator Adapters
    def langsmith_correctness_evaluator(run, example) -> dict:
        q = example.inputs["question"]
        ref = example.outputs["answer"]
        pred = run.outputs["answer"]
        grade = evaluate_correctness(q, ref, pred)
        return {"key": "correctness", "score": grade.score, "comment": grade.reason}

    def langsmith_boundary_evaluator(run, example) -> dict:
        ref = example.outputs["answer"]
        pred = run.outputs["answer"]
        score = evaluate_exact_boundary(ref, pred)
        return {"key": "boundary_refusal", "score": score}

    # 4. Run Experiment via LangSmith Evaluation SDK
    print("Running evaluation experiment against LangSmith...")
    results = evaluate(
        target_application,
        data=DATASET_NAME,
        evaluators=[langsmith_correctness_evaluator, langsmith_boundary_evaluator],
        experiment_prefix="rag-v1-baseline",
        max_concurrency=2,
    )
    print("LangSmith evaluation complete. Inspect your experiment results on the web UI.")


def main():
    run_local_evaluation()
    run_langsmith_experiment()


if __name__ == "__main__":
    main()