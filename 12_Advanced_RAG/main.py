import uuid
from typing import Any, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks.manager import (
    CallbackManagerForRetrieverRun,  # FIX 1: added missing import
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field  # FIX 2: added ConfigDict import
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

# =====================================================================
# 1. Model & Vector Store Configuration
# =====================================================================
LLM_MODEL = "gemma4:26b"
EMBED_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "advanced_rag_production"

llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://localhost:11434")
qdrant_client = QdrantClient(url=QDRANT_URL)


# =====================================================================
# 2. Pydantic Schemas for Query Transformation & Compression
# =====================================================================
class QueryExpansion(BaseModel):
    rewritten_query: str = Field(
        description="Standalone search query with pronouns and ambiguities resolved."
    )
    multi_queries: list[str] = Field(
        description="2-3 alternative search queries focusing on different sub-aspects or synonyms."
    )


class CompressedContext(BaseModel):
    relevant_facts: list[str] = Field(
        description="List of extracted sentences/facts directly answering the query, omitting irrelevant fluff."
    )


# =====================================================================
# 3. Parent-Child Ingestion Engine
# =====================================================================
class ParentChildStore:
    """Manages full parent documents in memory while pushing child chunks to vector/sparse stores."""

    def __init__(self):
        self.parent_docs: dict[str, Document] = {}

    def ingest(
        self,
        raw_documents: list[Document],
        vector_store: QdrantVectorStore,
    ) -> list[Document]:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)

        all_child_docs: list[Document] = []

        for raw_doc in raw_documents:
            # 1. Create Parent Documents
            parents = parent_splitter.split_documents([raw_doc])
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata["parent_id"] = parent_id
                self.parent_docs[parent_id] = parent

                # 2. Create Child Documents linked to Parent ID
                children = child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["parent_id"] = parent_id
                    # Propagate tenant and security metadata
                    child.metadata["tenant_id"] = parent.metadata.get("tenant_id")
                    child.metadata["source"] = parent.metadata.get("source")
                    all_child_docs.append(child)

        # 3. Upsert children to Qdrant
        vector_store.add_documents(all_child_docs)
        return all_child_docs

    def resolve_parents(self, child_docs: list[Document]) -> list[Document]:
        """Resolves unique parent documents from matched child chunks."""
        seen_parents: set[str] = set()
        resolved: list[Document] = []

        for child in child_docs:
            p_id = child.metadata.get("parent_id")
            if p_id and p_id in self.parent_docs and p_id not in seen_parents:
                seen_parents.add(p_id)
                resolved.append(self.parent_docs[p_id])

        return resolved


# =====================================================================
# 4. Advanced Hybrid + MMR + RRF Retriever
# =====================================================================
class AdvancedHybridRetriever(BaseRetriever):
    """
    Orchestrates:
    - Multi-Query Expansion
    - Dense MMR Retrieval (via Qdrant with Metadata Filtering)
    - Sparse BM25 Retrieval
    - Reciprocal Rank Fusion (RRF)
    - Parent Document Resolution
    """

    # FIX 3: Allow non-Pydantic types (ParentChildStore, LCEL chain) as fields.
    # Without this, Pydantic v2 raises a PydanticUserError at class definition time.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: QdrantVectorStore
    bm25_retriever: BM25Retriever
    parent_store: ParentChildStore
    query_expander: Any
    tenant_id: str
    top_k: int = 4

    # FIX 4: Added required `*, run_manager` parameter.
    # LangChain's BaseRetriever declares _get_relevant_documents as an abstract method
    # with this exact signature. Omitting run_manager causes a TypeError when the
    # retriever is invoked via .get_relevant_documents() or inside an LCEL chain.
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Step A: Query Rewriting + Multi-Query Generation
        expansion: QueryExpansion = self.query_expander.invoke({"question": query})
        all_queries = [expansion.rewritten_query] + expansion.multi_queries

        # Step B: Tenant Filter for Qdrant
        tenant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.tenant_id",
                    match=qmodels.MatchValue(value=self.tenant_id),
                )
            ]
        )

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        rrf_constant = 60

        # Step C: Multi-Query Search across Dense (MMR) & Sparse (BM25)
        for q in all_queries:
            # 1. Dense MMR Search (Balances relevance vs chunk diversity)
            dense_docs = self.vector_store.max_marginal_relevance_search(
                query=q,
                k=4,
                fetch_k=15,
                lambda_mult=0.6,
                filter=tenant_filter,
            )

            # 2. Sparse BM25 Search
            sparse_docs = self.bm25_retriever.invoke(q)
            # Filter BM25 results by tenant_id in-memory
            sparse_docs = [
                d for d in sparse_docs if d.metadata.get("tenant_id") == self.tenant_id
            ][:4]

            # 3. Reciprocal Rank Fusion on Child Chunks
            for rank, doc in enumerate(dense_docs):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

            for rank, doc in enumerate(sparse_docs):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Sort child chunks by descending RRF score
        sorted_child_chunks = [
            doc_map[doc_id]
            for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Step D: Parent Resolution (Fetch broader parent context)
        parent_documents = self.parent_store.resolve_parents(sorted_child_chunks)
        return parent_documents[: self.top_k]


# =====================================================================
# 5. Contextual Compression & Reranker Pipeline
# =====================================================================
def build_contextual_compressor():
    """Extracts only query-relevant facts from parent docs to eliminate context noise."""
    compression_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise context extraction engine. Given a user query and a source text, "
                "extract ONLY the sentences or facts that directly answer or provide context for the query. "
                "Do not summarize or invent new facts. If no sentences are relevant, return an empty list.",
            ),
            (
                "human",
                "User Query: {query}\n\nSource Document:\n{document_text}",
            ),
        ]
    )
    return compression_prompt | llm.with_structured_output(CompressedContext)


def build_advanced_rag_chain(
    retriever: AdvancedHybridRetriever,
    compressor: Any,
):
    def compress_and_rerank(inputs: dict[str, Any]) -> str:
        query = inputs["question"]
        parent_docs: list[Document] = inputs["documents"]

        compressed_passages: list[str] = []

        for doc in parent_docs:
            extracted: CompressedContext = compressor.invoke(
                {"query": query, "document_text": doc.page_content}
            )
            if extracted.relevant_facts:
                passage_text = " ".join(extracted.relevant_facts)
                source_tag = doc.metadata.get("source", "unknown")
                compressed_passages.append(f"[Source: {source_tag}]\n{passage_text}")

        if not compressed_passages:
            return "No relevant context found."

        return "\n\n---\n\n".join(compressed_passages)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert systems engineer. Answer the user question strictly using the provided context. "
                "Always cite the source document name when asserting facts. "
                "If the context is insufficient, state clearly: 'Insufficient verified context available.'\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # Master LCEL Execution Graph
    return (
        {
            "documents": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | RunnablePassthrough.assign(
            context=RunnableLambda(compress_and_rerank)
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )


# =====================================================================
# 6. Sample Dataset & Verification Runner
# =====================================================================
def get_sample_corpus() -> list[Document]:
    return [
        Document(
            page_content=(
                "PostgreSQL uses Multiversion Concurrency Control (MVCC) to handle high concurrency. "
                "Instead of locking rows for reading, each transaction sees a consistent snapshot of the data. "
                "Write-Ahead Logging (WAL) ensures ACID durability by writing changes to disk before updating data files. "
                "In high-load systems, table bloat occurs when dead tuples are not vacuumed fast enough, "
                "requiring aggressive autovacuum tuning and pg_repack maintenance."
            ),
            metadata={"source": "postgres_internals.pdf", "tenant_id": "tenant_alpha"},
        ),
        Document(
            page_content=(
                "FastAPI executes standard 'def' endpoints in an external anyio worker threadpool to prevent blocking the main thread. "
                "Conversely, 'async def' endpoints run directly on the event loop, meaning CPU-heavy or blocking synchronous I/O operations "
                "will stall all concurrent requests. Always use async database drivers like asyncpg with SQLAlchemy async sessions."
            ),
            metadata={"source": "fastapi_concurrency.md", "tenant_id": "tenant_alpha"},
        ),
        Document(
            page_content=(
                "Tenant Beta Confidential Security Protocol: "
                "All JWT access tokens must be signed using RS256 with 4096-bit asymmetric keys rotated every 30 days. "
                "Refresh tokens must be stored as SHA-256 hashes in Redis with a strict 7-day TTL and single-use revocation."
            ),
            metadata={"source": "beta_security.md", "tenant_id": "tenant_beta"},
        ),
    ]


def main():
    print("=== Initializing Advanced RAG Components ===")

    # Step 1: Determine vector dimension dynamically from the embedding model
    sample_vector = embeddings.embed_query("probe dimension")
    vector_size = len(sample_vector)

    # Step 2: Reset and explicitly create the Qdrant collection with vector configuration
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Step 3: Attach vector store wrapper to the provisioned collection
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Step 4: Ingest Parent-Child Documents
    parent_store = ParentChildStore()
    raw_docs = get_sample_corpus()
    child_docs = parent_store.ingest(raw_docs, vector_store)

    # Step 5: Initialize Sparse BM25 Index over child chunks
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = 4

    # Step 6: Build Query Transformation Subchain
    expansion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert search engine query optimizer. "
                "Given a conversational user query, rewrite it into a single clean keyword search query, "
                "and generate 2 alternative perspectives/sub-queries.",
            ),
            ("human", "{question}"),
        ]
    )
    query_expander = expansion_prompt | llm.with_structured_output(QueryExpansion)

    # Step 7: Assemble Advanced Hybrid Retriever for 'tenant_alpha'
    advanced_retriever = AdvancedHybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        parent_store=parent_store,
        query_expander=query_expander,
        tenant_id="tenant_alpha",
        top_k=2,
    )

    # Step 8: Assemble Compression & Master Pipeline
    compressor = build_contextual_compressor()
    rag_pipeline = build_advanced_rag_chain(advanced_retriever, compressor)

    # =====================================================================
    # Test Suite: Multi-Feature Verification
    # =====================================================================
    print("\n--- Test 1: Full Query Execution (Rewriting + Multi-Query + Hybrid + Parent + Compression) ---")
    query_1 = "Why does autovacuum tuning matter when dealing with dead tuples in Postgres?"
    print(f"User Query: {query_1}")
    response_1 = rag_pipeline.invoke({"question": query_1})
    print(f"\nFinal Attributed Response:\n{response_1}\n")

    print("\n--- Test 2: Security & Tenant Isolation Check ---")
    # 'tenant_alpha' queries for information stored exclusively under 'tenant_beta'
    query_2 = "What are the token rotation and asymmetric key signing rules?"
    print(f"Tenant Alpha Querying Beta Data: {query_2}")
    response_2 = rag_pipeline.invoke({"question": query_2})
    print(f"\nResponse (Expected Guardrail):\n{response_2}\n")


if __name__ == "__main__":
    main()