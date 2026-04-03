# Ecom ChatBot — Text-to-SQL AI Assistant

A production-ready **conversational AI agent** that converts natural language questions into SQL queries against a PostgreSQL ecommerce database. Built with **LangGraph** for stateful multi-turn conversations, **RAG** (ChromaDB) for schema retrieval, and **FastAPI** for serving.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Pipeline                  │
│                                                             │
│  FollowUp        Rewriter      Orchestrator                 │
│  Detector  ──►  (if needed) ──►  (state reset)             │
│                                      │                      │
│                              Intent Classifier              │
│                                      │                      │
│                              Schema Retriever (RAG)         │
│                                      │                      │
│                              Schema Summarizer              │
│                                      │                      │
│                              Query Planner                  │
│                                      │                      │
│                         ┌────────────┴────────────┐        │
│                         │   SQL Loop (per step)   │        │
│                         │  SQL Agent → Validator  │        │
│                         │  → Executor → Router    │        │
│                         └────────────┬────────────┘        │
│                                      │                      │
│                              Analytics Reporter             │
│                                      │                      │
│                              Memory Agent                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Natural Language Response
```

### Key Design Decisions

- **Multi-step planning** — complex queries are broken into sequential steps; each step's result is available to subsequent steps.
- **Schema RAG** — table DDL and FK relationships are embedded in ChromaDB (`BAAI/bge-small-en-v1.5`). Retrieval uses vector similarity + keyword re-ranking + FK graph expansion to always include joined tables.
- **Multi-turn sessions** — thread IDs map to LangGraph `InMemorySaver` checkpoints. Follow-up questions are detected and rewritten as standalone queries before pipeline execution.
- **Defense-in-depth SQL safety** — hard-blocked DML/DDL keywords (`DELETE`, `DROP`, `INSERT`, `UPDATE`, `TRUNCATE`, `GRANT`, `ALTER`) enforced before any LLM validation step. A separate `restricted_operations` intent blocks business-policy violations.

---

## Project Structure

```
Ecom ChatBot/
├── core/
│   ├── state.py              # AgentState (shared state across all nodes)
│   ├── settings.py           # Pydantic-settings config (.env loader)
│   ├── prompt_registry.py    # Singleton prompt cache
│   └── prompt_loader.py      # YAML prompt loader
├── pipeline/
│   ├── graph.py              # LangGraph state machine definition
│   ├── draw_graph.py         # Exports graph.png visualization
│   ├── nodes/                # 12 pipeline nodes
│   └── edges/                # Conditional routing functions
├── schema/                   # Pydantic output schemas for all LLM calls
├── prompts/                  # YAML prompt templates (system + user messages)
├── llm/                      # LLM client wrappers (OpenAI, Groq)
├── evaluation/               # Ragas + custom retrieval evaluation
├── tests/
│   ├── unit/                 # Pure unit tests (no LLM, no DB)
│   ├── behavioral/           # LLM-backed integration tests
│   ├── integration/          # Live DB + ChromaDB tests
│   └── fixtures/             # Shared AgentState builders
├── schema_rag_pipeline.py    # RAG core: embedding, retrieval, FK expansion
├── main.py                   # FastAPI application entry point
├── docker-compose.yml
└── Dockerfile
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- PostgreSQL database (see [Database Setup](#database-setup))
- API keys for OpenAI, Groq, and LangSmith

### 1. Install dependencies

```bash
uv sync          # preferred
# or
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your keys and DB URLs
```

Required variables:

```env
OPENAI_API_KEY=
GROQ_API_KEY=
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/ecommerce
READ_DATABASE_URL=postgresql+psycopg://readonly_user:pass@host:5432/ecommerce
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=Ecommerce RAG
LANGSMITH_TRACING=true
```

### 3. Initialize ChromaDB embeddings (one-time)

```bash
python schema_rag_pipeline.py
```

This embeds all table schemas, descriptions, and sample queries into the local `chroma_db/` directory.

### 4. Start the server

```bash
python main.py                        # Production
uvicorn main:api --reload             # Dev mode with hot-reload
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## API Reference

### `POST /invoke`

Synchronous query execution. Blocks until the full pipeline completes.

```json
// Request
{
  "thread_id": 1,
  "user_query": "What are the top 5 cities by order volume this month?"
}

// Response
{
  "thread_id": 1,
  "user_query": "What are the top 5 cities by order volume this month?",
  "agent_output": "Mumbai leads with 1,200 orders, followed by Delhi (950)..."
}
```

### `POST /invoke/stream`

Server-Sent Events (SSE) stream. Each event contains the state updates from one pipeline node.

```
event: message
data: {"FOLLOW_UP DETECTOR": {"followup": {"is_followup": "false", "type": "new"}}}

event: message
data: {"INTENT_CLASSIFIER": {"intent_result": {"intent": "top_n", ...}}}

...

event: done
data: [DONE]
```

### `GET /messages/{thread_id}`

Returns the conversation history for a thread.

```json
{
  "thread_id": 1,
  "messages": [
    {"role": "human", "content": "What are the top 5 cities..."},
    {"role": "assistant", "content": "Mumbai leads with 1,200 orders..."}
  ]
}
```

### `GET /sessions`

Lists all active thread IDs.

```json
{ "sessions": [1, 2, 3] }
```

---

## Pipeline Nodes

| Node | Model | Purpose |
|---|---|---|
| `followup_detector` | Groq (Llama 3.1 8B) | Detects whether the query is a follow-up, and what type (refine, drilldown, explain, etc.) |
| `followup_rewriter` | OpenAI (gpt-4o-mini) | Rewrites follow-up queries as fully self-contained questions using prior context |
| `orchestrator` | — | Resets all per-turn state fields before each fresh execution |
| `intent_classifier` | Groq (Llama 3.1 8B) | Classifies into 9 intent types: `kpi_lookup`, `trend_analysis`, `comparison`, `top_n`, `anomaly`, `forecast`, `segmentation`, `operational`, `restricted_operations` |
| `schema_retriever` | ChromaDB (RAG) | Retrieves relevant table schemas via vector similarity + FK graph expansion |
| `schema_summarizer` | OpenAI (gpt-4o-mini) | Condenses retrieved schema DDL to only columns relevant to the current question |
| `query_planner` | OpenAI (gpt-4o-mini) | Breaks complex questions into ordered, individually-executable plan steps |
| `sql_agent` | OpenAI (gpt-4o-mini) | Generates SQL for the current plan step; receives validator feedback on retries |
| `validator` | OpenAI (gpt-4o-mini) + rules | Hard-blocks DML/DDL; LLM checks join correctness, column existence, aggregation logic |
| `sql_executor` | Postgres | Executes SQL via `pandas.read_sql_query()` against the read-only database connection |
| `step_router` | — | Advances `plan_index`, saves step results, resets retry counter |
| `analytics_reporter` | OpenAI (gpt-4o-mini) | Generates a natural language narrative from all step results |
| `memory_agent` | — | Persists `last_objective`, `last_result_summary` for the next conversation turn |

### Retry & Error Handling

```
SQL Agent → Validator
               │
         ┌─────┴──────┐
       valid        invalid
         │             │
      Executor    SQL Agent (retry with repair_hint)
                        │
                  [max 3 retries] → Memory Agent → END (failure response)
```

---

## Schema RAG Pipeline

`schema_rag_pipeline.py` is the core retrieval engine:

1. **Embedding** — Each table is represented as a document combining its DDL, human-readable description, column semantics, and sample queries. Embedded using `BAAI/bge-small-en-v1.5`.
2. **Retrieval** — Top-6 tables by cosine similarity, then re-ranked using `TABLE_KEYWORD_SIGNALS` (domain-specific boost/penalty rules).
3. **FK Expansion** — A deterministic graph walk adds any tables reachable via foreign key edges from the top-ranked tables, preventing missed joins.
4. **Context Formatting** — Final output is a set of `CREATE TABLE` statements fed directly to the SQL agent.

To re-embed after schema changes:

```bash
python schema_rag_pipeline.py
```

---

## Prompts

All LLM prompts live in `prompts/` as YAML files with `system` and `user` fields. They are loaded at startup by `core/prompt_registry.py` (singleton) and resolved via `core/prompt_loader.py`.

To edit a prompt, modify the YAML file — no code changes required. The prompt loader resolves `{template_variable}` placeholders at call time.

---

## Testing

```bash
# Unit tests only (no LLM, no DB required)
pytest tests/unit/

# Skip live DB tests
pytest -m "not integration"

# Skip non-deterministic LLM evaluation tests
pytest -m "not llm_eval"

# All tests
pytest
```

### Test Categories

| Category | Location | Requires |
|---|---|---|
| Unit | `tests/unit/` | Nothing (fully mocked) |
| Behavioral | `tests/behavioral/` | LLM API keys |
| Integration | `tests/integration/` | Postgres + ChromaDB |
| RAG Evaluation | `evaluation/` | LLM + DB + Ragas |

---

## Docker

```bash
# Build and run
docker build -t ecom-chatbot .
docker run -p 8000:8000 --env-file .env ecom-chatbot

# With docker-compose
docker-compose up
```

The Docker image pre-caches the `BAAI/bge-small-en-v1.5` embedding model at build time. Mount your pre-built ChromaDB:

```yaml
volumes:
  - ./chroma_db:/app/chroma_db
```

---

## Database Setup

The ecommerce PostgreSQL schema includes ~25 tables:

```
users, orders, order_items, products, product_variants,
categories, brands, payments, shipments, reviews,
inventory, inventory_logs, review_votes, shipment_tracking, ...
```

Initialization scripts are in `docker/postgres/init/`. Two database users are expected:

- **`DATABASE_URL`** — write-capable user (used only for schema inspection at startup)
- **`READ_DATABASE_URL`** — read-only analyst user (all SQL queries execute here)

---

## Observability

All LLM calls are traced via **LangSmith**. Set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` to see full traces including token counts, latencies, and intermediate states per node.

---

## Visualize the Agent Graph

```bash
python pipeline/draw_graph.py   # Outputs graph.png
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| Web framework | FastAPI |
| LLM (SQL / planning / reporting) | OpenAI gpt-4o-mini |
| LLM (classification / detection) | Groq Llama-3.1-8B-Instant |
| Embeddings | HuggingFace BAAI/bge-small-en-v1.5 |
| Vector store | ChromaDB |
| Database | PostgreSQL (psycopg3) |
| Data processing | Pandas + SQLAlchemy |
| Config | Pydantic Settings |
| Tracing | LangSmith |
| RAG evaluation | Ragas + DeepEval |
| Testing | Pytest |
| Packaging | uv |
