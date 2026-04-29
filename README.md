# Ecom ChatBot — Conversational Text-to-SQL AI Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.135+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-Stateful%20Agent-FF6B35?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<p align="center">
  A production-ready AI agent that converts natural language questions into SQL queries against a PostgreSQL e-commerce database — with multi-turn memory, schema RAG, multi-step planning, and defense-in-depth SQL safety.
</p>

---

## Demo

<!-- TODO: Replace this block with your demo video once recorded -->
> **Video walkthrough coming soon.**
>
> [![Demo Video Placeholder](https://via.placeholder.com/800x450/1a1a2e/ffffff?text=Demo+Video+Coming+Soon+-+Click+to+Watch)](https://github.com/satyasai-1/ecom-chatbot)
<!-- Replace the image above with a real thumbnail and link once the video is uploaded, e.g.:
[![Watch the demo](docs/demo_thumbnail.png)](https://youtu.be/YOUR_VIDEO_ID)
-->

---

## Features

- **Natural language to SQL** — ask questions in plain English; the agent writes, validates, and executes SQL for you
- **Multi-turn conversations** — thread-based sessions with LangGraph `InMemorySaver` checkpoints; follow-up questions are automatically detected and rewritten as standalone queries
- **Multi-step query planning** — complex questions are decomposed into ordered plan steps; each step's result feeds the next
- **Schema RAG** — table DDL and foreign-key relationships are embedded in ChromaDB; retrieval uses vector similarity + keyword re-ranking + FK graph expansion
- **Defense-in-depth SQL safety** — hard-blocked DML/DDL keywords enforced before any LLM step; a `restricted_operations` intent gate blocks policy violations
- **Structured LLM outputs** — every node uses Pydantic schemas with `llm.with_structured_output()` for reliable, type-safe results
- **Full observability** — all LLM calls traced in LangSmith with token counts, latencies, and per-node intermediate states
- **Streaming API** — Server-Sent Events endpoint streams each pipeline node's output in real time
- **Comprehensive test suite** — unit, behavioral (LLM-backed), integration, and GEval evaluation tiers

---

## Architecture

### Agent Pipeline

<!-- TODO: Replace the placeholder below with your generated LangGraph agent diagram -->
run `python pipeline/draw_graph.py` to generate `pipeline/graph.png` locally._
>
> ![Agent Pipeline Graph](pipeline/graph.png)
<!-- Once you have a cleaner diagram, swap the path above, e.g.:
![Agent Pipeline Graph](docs/agent_pipeline.png)
-->

**Execution flow:**

```
START
  → followup_detector    (is this a follow-up or explanation request?)
  → followup_rewriter    (rewrite into a self-contained resolved_query)
  ↓ [rewriter_router: reuse → analytics_reporter | rewrite/fresh → orchestrator]
  → orchestrator         (reset per-turn execution state)
  → intent_classifier    (classify into 9 intent types)
  → schema_retriever     (RAG: fetch relevant table schemas from ChromaDB)
  → schema_summarizer    (LLM: compress retrieved schema to relevant columns)
  → query_planner        (LLM: break query into ordered executable steps)
  ↺ [loop per step]:
      → sql_agent        (LLM: generate SQL for current step)
      → validator        (hard rules + LLM: validate safety & correctness)
      → sql_executor     (run SQL against PostgreSQL read-only connection)
      → step_router      (advance plan or loop back if more steps remain)
  → analytics_reporter   (LLM: generate natural language narrative from results)
  → memory_agent         (persist context for follow-up queries)
END
```

### Retry & Error Handling

```
SQL Agent ──► Validator
                  │
           ┌──────┴──────┐
         valid         invalid
           │               │
        Executor    SQL Agent (retry with repair_hint)
                           │
                     [max 3 retries]
                           │
                    Memory Agent → END (graceful failure response)
```

---

## Pipeline Nodes

| Node | Model | Purpose |
|---|---|---|
| `followup_detector` | Groq Llama-3.1-8B | Detects follow-ups and classifies type (refine, drilldown, explain, etc.) |
| `followup_rewriter` | gpt-4o-mini | Rewrites context-dependent follow-ups into fully self-contained questions |
| `orchestrator` | — | Resets all per-turn state fields before each fresh execution |
| `intent_classifier` | Groq Llama-3.1-8B | Classifies into 9 intent types (see below) |
| `schema_retriever` | ChromaDB RAG | Vector similarity + FK graph expansion to retrieve relevant table schemas |
| `schema_summarizer` | gpt-4o-mini | Condenses retrieved DDL to only the columns relevant to the current question |
| `query_planner` | gpt-4o-mini | Breaks complex questions into ordered, individually-executable plan steps |
| `sql_agent` | gpt-4o-mini | Generates SQL per plan step; incorporates validator repair hints on retries |
| `validator` | gpt-4o-mini + rules | Hard-blocks DML/DDL; LLM checks join correctness, column existence, aggregation |
| `sql_executor` | PostgreSQL | Executes SQL via `pandas.read_sql_query()` on a read-only database connection |
| `step_router` | — | Advances `plan_index`, persists step results, resets retry counter |
| `analytics_reporter` | gpt-4o-mini | Synthesizes a natural language narrative from all plan step results |
| `memory_agent` | — | Persists `last_objective` and `last_result_summary` for the next conversation turn |

**Intent types (9):** `kpi_lookup` · `trend_analysis` · `comparison` · `top_n` · `anomaly` · `forecast` · `segmentation` · `operational` · `restricted_operations`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (stateful multi-node graph) |
| Web framework | FastAPI + Server-Sent Events |
| LLM — SQL / planning / reporting | OpenAI `gpt-4o-mini` |
| LLM — classification / detection | Groq `llama-3.1-8b-instant` |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Vector store | ChromaDB |
| Database | PostgreSQL 16 (psycopg3 + SQLAlchemy) |
| Data processing | Pandas |
| Output validation | Pydantic v2 + `with_structured_output` |
| Config | Pydantic Settings |
| Tracing & observability | LangSmith |
| RAG evaluation | Ragas + DeepEval (GEval) |
| Testing | Pytest (unit / behavioral / integration / eval) |
| Packaging | uv |
| Containerization | Docker (multi-stage) + Docker Compose |

---

## Project Structure

```
Ecom ChatBot/
├── core/
│   ├── state.py              # AgentState — shared state passed between all nodes
│   ├── settings.py           # Pydantic-settings config (.env loader)
│   ├── prompt_registry.py    # Singleton prompt cache (loaded once at startup)
│   └── prompt_loader.py      # YAML prompt loader with template variable resolution
├── pipeline/
│   ├── graph.py              # LangGraph state machine definition
│   ├── draw_graph.py         # Exports graph.png visualization
│   ├── nodes/                # 12 pipeline node implementations
│   └── edges/                # Conditional routing functions
├── schema/                   # Pydantic output schemas for all LLM calls
├── prompts/                  # Versioned YAML prompt templates (system + user messages)
├── llm/                      # LLM client wrappers (OpenAI, Groq)
├── datasets/                 # Annotated JSON evaluation datasets (per node)
├── evaluation/               # Ragas retrieval evaluation + results
├── tests/
│   ├── unit/                 # Pure logic tests — no LLM, no DB
│   ├── behavioral/           # LLM-backed accuracy tests against labeled datasets
│   ├── integration/          # Live DB + ChromaDB tests
│   ├── evaluation/           # DeepEval GEval per-node quality scoring
│   └── fixtures/             # Shared AgentState builders
├── docker/postgres/init/     # DB init scripts (roles, schema, seed data)
├── schema_rag_pipeline.py    # RAG core: embedding, retrieval, FK expansion
├── main.py                   # FastAPI application entry point
├── docker-compose.yml
└── Dockerfile
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- PostgreSQL database (or use Docker Compose — it provisions one automatically)
- API keys: OpenAI, Groq, LangSmith

### 1. Install dependencies

```bash
uv sync          # preferred
# or
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys and database URLs
```

```env
OPENAI_API_KEY=
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
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

This embeds all table DDL, column semantics, and sample queries into the local `chroma_db/` directory using `BAAI/bge-small-en-v1.5`.

### 4. Start the server

```bash
python main.py                    # Production
uvicorn main:api --reload         # Dev mode with hot-reload
```

API available at `http://localhost:8000` · Swagger docs at `http://localhost:8000/docs`

### Docker (recommended)

```bash
docker compose up --build         # Starts app + PostgreSQL
docker compose down -v            # Tear down (removes volume)
```

The Docker image pre-caches the `BAAI/bge-small-en-v1.5` model at build time, avoiding a 30–60 s cold-start on first run.

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

Server-Sent Events stream. Each event contains state updates from one pipeline node.

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

Returns the full conversation history for a thread.

```json
{
  "thread_id": 1,
  "messages": [
    { "role": "human", "content": "What are the top 5 cities...?" },
    { "role": "assistant", "content": "Mumbai leads with 1,200 orders..." }
  ]
}
```

### `GET /sessions`

Lists all active thread IDs.

```json
{ "sessions": [1, 2, 3] }
```

---

## Schema RAG Pipeline

`schema_rag_pipeline.py` powers schema-aware SQL generation:

1. **Embedding** — Each table is represented as a document combining its DDL, human-readable description, column semantics, and sample queries. Embedded with `BAAI/bge-small-en-v1.5`.
2. **Retrieval** — Top-6 tables by cosine similarity, then re-ranked using `TABLE_KEYWORD_SIGNALS` (domain-specific boost/penalty rules).
3. **FK Expansion** — A deterministic graph walk adds any tables reachable via foreign-key edges from the top-ranked tables, preventing missed joins.
4. **Context Formatting** — Final output is a set of `CREATE TABLE` statements passed directly to the SQL agent.

To re-embed after schema changes:

```bash
python schema_rag_pipeline.py
```

---

## Prompts

All LLM prompts live in `prompts/` as versioned YAML files with `system` and `user` message arrays. They are loaded at startup by `core/prompt_registry.py` (singleton) and resolved via `core/prompt_loader.py`.

To swap a prompt version, update the path in `core/prompt_registry.py` — no other code changes required.

**Active versions:** `validation_prompt_v5.yaml` · `nl_response_prompt_v3.yaml` · `schema_summarizer_prompt_v5.yaml`

---

## Testing

```bash
pytest tests/unit/                        # Unit tests (no LLM, no DB)
pytest -m behavioral tests/behavioral/    # LLM-backed accuracy tests
pytest -m integration tests/integration/  # Requires live DB + ChromaDB
pytest -m llm_eval tests/evaluation/      # DeepEval GEval (slow, costs tokens)
```

| Tier | Location | Requires |
|---|---|---|
| Unit | `tests/unit/` | Nothing (fully mocked) |
| Behavioral | `tests/behavioral/` | LLM API keys |
| Integration | `tests/integration/` | PostgreSQL + ChromaDB |
| Evaluation (GEval) | `tests/evaluation/` | LLM API keys — non-deterministic |
| RAG Evaluation | `evaluation/` | LLM + DB + Ragas |

---

## Database

The PostgreSQL schema includes ~25 tables across the full e-commerce domain:

```
users, orders, order_items, products, product_variants,
categories, brands, payments, shipments, reviews,
inventory, inventory_logs, review_votes, shipment_tracking, ...
```

Two database roles are expected:

| Variable | Role | Used by |
|---|---|---|
| `DATABASE_URL` | Write-capable user | Schema inspection at startup only |
| `READ_DATABASE_URL` | `analyst_bot` (SELECT-only) | All SQL query execution |

Init scripts in `docker/postgres/init/` run in order on first container start: `00_roles.sql` (creates `analyst_bot`) → `01_dump.sql` (schema + seed data).

---

## Observability

All LLM calls are traced via **LangSmith** (project: `Ecommerce RAG`). Set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` to see full traces including token counts, latencies, and intermediate state per node.

To visualize the agent graph locally:

```bash
python pipeline/draw_graph.py   # Outputs pipeline/graph.png
```

---
