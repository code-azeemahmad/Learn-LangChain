# lesson20/main.py
from contextlib import asynccontextmanager

from app.ai import create_ai_agent, init_vector_store
from app.api import router as chat_router
from app.service import ChatService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

# Configuration
MODEL_NAME = "gemma4:26b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fastapi_langchain_integration"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown resource lifecycle."""
    print(" [Startup] Initializing Qdrant connection and Agent Graph...")

    # 1. Initialize Vector Store
    vector_store = init_vector_store(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )

    # 2. Compile Stateful Agent Engine
    agent_graph = create_ai_agent(
        model_name=MODEL_NAME,
        vector_store=vector_store,
    )

    # 3. Attach ChatService to Application State
    app.state.chat_service = ChatService(agent_graph=agent_graph)
    app.state.qdrant_client = QdrantClient(url=QDRANT_URL)
    print(" [Startup] ChatService ready.")

    yield

    print(" [Shutdown] Closing connections...")
    app.state.qdrant_client.close()


app = FastAPI(
    title="Enterprise AI Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "chat_engine"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)