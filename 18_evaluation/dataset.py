# lesson19/dataset.py
from typing import Any

# Curated benchmark dataset
EVALUATION_DATASET: list[dict[str, Any]] = [
    {
        "inputs": {"question": "What is FastAPI?"},
        "outputs": {
            "answer": "FastAPI is a modern, high-performance web framework for building APIs with Python and standard type hints."
        },
        "metadata": {"category": "definition", "difficulty": "easy"},
    },
    {
        "inputs": {"question": "How does FastAPI handle dependency injection?"},
        "outputs": {
            "answer": "FastAPI handles dependency injection using the Depends function to share database sessions, authentication, and security logic."
        },
        "metadata": {"category": "architecture", "difficulty": "medium"},
    },
    {
        "inputs": {"question": "What server is commonly used to execute FastAPI applications?"},
        "outputs": {
            "answer": "FastAPI applications are commonly executed using an ASGI server such as Uvicorn."
        },
        "metadata": {"category": "deployment", "difficulty": "easy"},
    },
    {
        "inputs": {"question": "How does PostgreSQL manage high write concurrency?"},
        "outputs": {
            "answer": "PostgreSQL utilizes Multiversion Concurrency Control (MVCC) and Write-Ahead Logging (WAL) to support concurrent transactions without read-locks."
        },
        "metadata": {"category": "database", "difficulty": "medium"},
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {
            "answer": "I do not have enough information to answer this based on the provided context."
        },
        "metadata": {"category": "boundary_out_of_domain", "difficulty": "hard"},
    },
]