## Lesson 21: Production LangChain
 
Shifts the question from *"How do I use LangChain?"* to *"How do I run it reliably in production?"*
 
### 1. Production Mental Model
 
```
CLIENT → FastAPI → Auth + Rate Limiting → Application Service
   → RAG / Agent / Tools → LangChain/LangGraph
   → Ollama / Qdrant / PostgreSQL → Observability (LangSmith)
```
 
The framework is one layer, not the whole system.
 
### 2. Don't Use LangChain Everywhere
 
Use it only where AI-specific abstractions add value. Keep business logic, DB access, auth, HTTP, and config in plain Python.
 
### 3. Control the Model Boundary
 
```
Application Service → Model abstraction → Provider → LLM
```
 
Centralize provider, model, temperature, timeout, retries, fallback via a settings object (e.g. `ModelSettings`).
 
### 4. Externalize Configuration
 
Use env vars (`LLM_PROVIDER`, `LLM_MODEL`, etc.) and a `settings` object, not hardcoded values.
 
### 5. Secrets
 
Never hardcode API keys, DB passwords, or LangSmith keys. Use `.env` / secret managers. **Source code holds structure, not secrets.**
 
### 6. Timeouts
 
Set explicit timeouts for every external dependency (LLM, Qdrant, HTTP, DB, agent execution) to avoid hung requests and resource buildup.
 
### 7–8. Retries
 
Retry transient failures (network blips, 429s, transient server errors). Don't retry permanent failures (auth errors, bad requests, invalid schema). Uncontrolled retries in agent loops multiply latency and cost — configure max retries, retryable errors, and backoff deliberately.
 
### 9. Fallback Models
 
Primary → fallback on failure improves availability, but fallbacks may differ in quality, tool-calling, structured output, context window, latency, and cost. Confirm the fallback supports what the workflow needs.
 
### 10. Provider Abstraction vs. LangChain Abstraction
 
A higher-level `ModelService` (Ollama / OpenAI / Gemini) can sit above LangChain for fallback, cost routing, and business policy — framework and application abstractions can coexist.
 
### 11. Cost Control
 
Track input/output tokens, model calls, tool calls, retrieval calls. Set per-request, per-user, and daily/monthly budgets, plus max agent iterations.
 
### 12. Agent Iteration Limits
 
Agents can loop indefinitely due to bad prompts, tool failures, or state bugs. Bound every agent's resource budget (LangChain middleware supports model-call limits).
 
### 13–14. Tool Security
 
Tools carry authority. Ask: who can call it, what can it access, can it mutate state, can it be undone, does it need approval? Classify tools:
- **Read-only** (lower risk): search, get, lookup
- **Mutating** (higher risk): create, send, delete, update, refund — require authorization, confirmation, audit logs, idempotency, human approval.
### 15–16. Prompt & Tool-Output Injection
 
User input, retrieved documents, and tool outputs can all contain embedded instructions. Treat external content as **data**, never as trusted instructions.
 
### 17. Authorization Happens Outside the Model
 
Never rely on system-prompt instructions as a security boundary. Enforce via:
```
JWT → authenticated identity → authorization → retrieval filter → Qdrant
JWT → authorization → tool access
```
 
### 18–19. Multi-Tenant & Conversation Isolation
 
Enforce `tenant_id` / `user_id` filters in retrieval. Always verify conversation ownership before using a client-provided `thread_id`.
 
### 20–21. Observability & Logging
 
LangSmith traces: latency, errors, calls, token usage, trace IDs — but redact private/sensitive data. Keep separate structured application logs (request_id, user_id, conversation_id, status) without full conversations, documents, or API keys.
 
### 22. Evaluation in Production
 
Combine offline evaluation (fixed dataset → score) with production evaluation (sampled real traffic) to catch quality regression, retrieval degradation, latency spikes, and agent loops.
 
### 23. Testing Strategy
 
- **Unit**: tools, retrievers, formatters, business rules
- **Integration**: Qdrant, PostgreSQL, Ollama, LangChain components
- **Workflow**: RAG chain, agent, state, tool selection
- **Evaluation**: answer/retrieval quality, regression
### 24. Determinism & Temperature
 
Lower temperature for query rewriting, classification, extraction, routing — but `temperature=0` doesn't guarantee full determinism (provider behavior, parallelism, sampling still vary). Keep evaluation suites.
 
### 25. Caching
 
Cache embeddings, retrieval results, deterministic transforms, expensive responses — but key caches by tenant/user/model/prompt-version/retrieval-config/question to avoid leaking one user's cached response to another.
 
### 26–27. Versioning
 
Version prompts and retrieval configuration (embedding model, chunk size/overlap, retriever type, top_k, reranker, query rewriting) so behavior changes are traceable.
 
### 28–29. Provider Failures & Graceful Degradation
 
Use distinguishable structured errors (`LLMUnavailable`, `RetrieverUnavailable`, etc.), not generic messages. Degrade gracefully where correctness isn't compromised (e.g., skip reranker); **fail closed** on security failures.
 
### 30. Agent Fallback Design
 
On tool failure: retry, fallback retriever, ask user, finish with uncertainty, or fail — never turn a retrieval failure into a confidently unsupported answer.
 
### 31. Human-in-the-Loop
 
Require approval for high-risk actions (financial transactions, deletions, external comms, permission changes) via LangGraph interrupt/state.
 
### 32. Production Agent Architecture
 
```
User → FastAPI → Auth → Authorization → Agent
  → Tool A/B/C → APIs / Qdrant / PostgreSQL
  → checkpoint → LangSmith
```
Insert human approval between agent and execution for mutating tools.
 
### 33–36. Scaling & Concurrency
 
Move beyond single-process setups. Consider connection pools, concurrency limits, backpressure, queueing, rate limits, horizontal scaling, and shared checkpoint storage. Avoid Python-global state (e.g. `active_conversations = {}`) in multi-worker deployments — use PostgreSQL, Redis, or a LangGraph checkpointer instead. Async improves I/O concurrency but doesn't remove infrastructure capacity limits — pair it with explicit concurrency control.
 
### 37. Security Checklist
 
Authentication · Authorization · Tenant isolation · User isolation · Tool authorization · Prompt injection · Tool-output injection · Secrets management · Input/output validation · Audit logs · Rate limiting · Data privacy · Observability privacy
 
### 38. Ownership Division
 
| Layer | Owns |
|---|---|
| LangChain | AI application abstractions |
| LangGraph | Stateful/agentic execution |
| FastAPI | HTTP/API layer |
| PostgreSQL | Application persistence |
| Qdrant | Vector retrieval/storage |
| Ollama/provider | Model execution |
 
### 39. Final Production Architecture
 
```
React → FastAPI → Auth + Validation → Application Service
  → RAG Workflow / Agent
  → Retriever (Qdrant + Reranker) / create_agent() (Model + Tools)
  → Ollama / Qdrant / PostgreSQL / External APIs
  → Checkpointer → LangGraph → LangSmith (Tracing + Evaluation)
```
 
### 40. Production Decision Framework
 
Before adopting LangChain for something, ask: What problem does it solve? What does it hide? Do we need that hidden behavior? Can we test it? How does it fail? How does it scale? Does it add unnecessary coupling?
 
### 41–42. When LangChain Helps vs. Isn't Needed
 
**Good fit**: multiple providers, structured outputs, retrieval, tools, reusable workflows, agents, streaming, observability, evaluation.
**Unnecessary**: single endpoint, single provider/model, simple prompt, plain text response — a direct SDK call may suffice.
 
### 43. When to Use LangGraph Directly
 
Go below `create_agent()` for custom state, branching, loops, interrupts, human approval, durable execution, or multi-agent orchestration.
 
```
LangChain abstractions → create_agent() → understand runtime → LangGraph directly (when needed)
```
 
### 44. Core Lesson
 
The real progression:
```
Fundamentals → Abstractions → Composition → State → Agents → Runtime → Production architecture
```
not framework-hopping.
 
### 45. Competency Checklist
 
Models (invoke/ainvoke/stream) · Messages (Human/AI/Tool) · Prompts (ChatPromptTemplate, MessagesPlaceholder) · Structured output (`with_structured_output`, Pydantic) · RAG (Document, Splitter, Embeddings, VectorStore, Retriever) · Composition (Runnable, RunnableSequence/Parallel/Passthrough, LCEL) · Agents (Tool, bind_tools, create_agent, LangGraph, State, Thread, Checkpoint, Streaming, Runtime) · Production (timeouts, retries, fallbacks, security, observability, evaluation, scaling)
 
---
 
## Curriculum Complete
 
All 21 core lessons are done. This does **not** mean "stop using LangChain" — the next phase is project-based.
 
### Broader Path
 
```
LangChain fundamentals → LangGraph runtime → Agent architectures
  → Production AI systems → MLOps → AI System Design
```
 
### Next Step: Build a Real System
 
**Production-Style Multi-User AI Knowledge Assistant**
 
```
React → FastAPI → JWT → multi-user/tenant isolation → document ingestion
  → Qdrant → RAG → LangChain → structured outputs → streaming → tools
  → create_agent → LangGraph state/checkpointing → LangSmith → evaluation
```
 
Rule: don't put LangChain everywhere — keep it inside a clean software architecture.
 
## Final Mental Model
 
```
                    AI APPLICATION
              ┌──────────┴──────────┐
        Deterministic            Agentic
          workflows              workflows
              │                     │
          LangChain             create_agent()
              │                     │
              │                LangGraph runtime
              └──────────┬──────────┘
                Models / Retrieval / Tools / Data
              ┌──────────┼──────────┐
           Ollama      Qdrant    PostgreSQL
                         │
                    LangSmith (tracing/evaluation)
```
 
**Rule:** understand the underlying operation, use framework abstractions where they add value, keep control of application-specific parts.