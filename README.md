# Agentic RAG Assistant

A document question-answering service where the model decides what to do next.
Instead of a fixed retrieve-then-generate pipeline, the model runs in a loop:
it searches the knowledge base, computes, asks the user for missing details, or
finalizes — and it explains each step.

Built with Python, FastAPI, ChromaDB, and Google Gemini.

## Highlights

- **Agentic research loop** — each step is a JSON decision
  (`search_knowledge_base`, `calculate_expression`, `get_current_datetime`,
  `ask_clarification`, `finalize`) chosen from a compacted evidence ledger.
- **Grounded answers** — retrieved chunks are bucketed into RELEVANT and
  MARGINAL by cosine distance, and a guardrail downgrades confidence when a
  search-driven answer has no relevant evidence.
- **Multi-turn clarification** — the agent can pause and ask a question, then
  resume the same session with the user's reply.
- **Bounded and observable** — hard step limits, output caps, per-query token
  accounting, estimated cost, a full action transcript, and failure reasons.
- **Production plumbing** — FastAPI backend, Streamlit UI, Docker Compose,
  retries with backoff, rate limiting, TTL response cache, and model fallback.
- **Evaluation harness** — nine scenarios including out-of-KB questions,
  multi-source synthesis, and injected tool failures.

## Architecture

```
                    ┌──────────────────┐
  browser ────────► │  Streamlit UI    │
                    │  src/prod/ui.py  │
                    └────────┬─────────┘
                             │ HTTP
                    ┌────────▼─────────┐
                    │  FastAPI backend │
                    │ src/prod/backend │
                    └───┬──────────┬───┘
                        │          │
             ┌──────────▼──┐   ┌───▼──────────────┐
             │  LLMService │   │  ResearchSession │
             │ retry/limit │   │  agentic loop    │
             └──────┬──────┘   └───┬──────────┬───┘
                    │              │          │
             ┌──────▼──────┐  ┌────▼─────┐ ┌──▼──────────┐
             │ Gemini API  │  │  Tools   │ │  VectorStore│
             │             │  │ calc/time│ │  (ChromaDB) │
             └─────────────┘  └──────────┘ └──────┬──────┘
                                                  │
                                          ┌───────▼────────┐
                                          │ Gemini embedder│
                                          └────────────────┘
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the loop protocol, prompts, and
failure handling in detail.

## Quickstart

Requirements: Python 3.11+ and a Google AI Studio API key.

```bash
git clone https://github.com/Sreinumder/agentic-rag-assistant.git
cd agentic-rag-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then add GOOGLE_API_KEY
```

Ingest the bundled sample documents and start chatting from the terminal:

```bash
python -m src.main
```

Inside the REPL:

```
/ingest docs          # chunk and embed the docs/ directory
/research <question>  # run the agentic research loop
/quit
```

### Backend + UI

```bash
# terminal 1
uvicorn src.prod.backend:app --reload --port 8000

# terminal 2
streamlit run src/prod/ui.py
```

Or with Docker Compose:

```bash
docker compose up --build
```

- API docs: <http://localhost:8000/docs>
- UI: <http://localhost:8501>

## API

| Method | Path       | Purpose                                                        |
| ------ | ---------- | -------------------------------------------------------------- |
| GET    | `/health`  | Liveness probe.                                                |
| GET    | `/status`  | Provider readiness, indexed chunks, cache size, model name.    |
| POST   | `/chat`    | Retrieve context and generate a response (cached by default).  |
| POST   | `/research`| Run the agent loop; send `reply` to answer a clarification.    |
| POST   | `/ingest`  | Chunk and embed a file or directory into the vector store.     |

Example:

```bash
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is supervised learning?"}'
```

When the response has `"status": "clarification_pending"`, repeat the request
with the same `message` plus `"reply": "..."`.

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Gemini API key (required). |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Primary generation model. |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | Fallback model. |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Chunking window. |
| `TOP_K` | `5` | Retrieved chunks per query. |
| `MAX_STEPS` | `5` | Hard cap on agent steps. |
| `RETRIEVAL_RELEVANT_MAX` | `0.42` | Cosine distance for "relevant". |
| `RETRIEVAL_MARGINAL_MAX` | `0.55` | Upper bound for "marginal". |
| `CACHE_TTL` / `CACHE_MAX_SIZE` | `300` / `1000` | Response cache. |
| `RATE_LIMIT_RPM` | `60` | Backend request budget. |
| `FAILURE_INJECTION` | `none` | Eval hook: `tool_unavailable`, `malformed_retrieval`, `max_steps`. |

## Evaluation

The harness in `eval/` runs the agent against a fixed query set and reports
completion, tool correctness, trajectory length, tokens, cost, and a
hard/soft/cascading failure taxonomy.

```bash
python eval/harness.py --report
```

A recorded run is in [`eval/RESULTS.md`](eval/RESULTS.md): **9/9 passed**,
average **2.89 steps**, **23,610 tokens**, about **$0.036** total on the
free-tier models.

## Tests

Unit tests run offline — no API key, network, or ChromaDB required:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Repository layout

```
src/
  agents/research_agent.py   agentic loop, ledger, guardrails
  agents/research_skill.md   full protocol, loaded on first tool use
  llm/gemini_client.py       Gemini wrapper with usage metadata
  rag/document_loader.py     file walking and chunking
  rag/embeddings.py          embedding client
  rag/vector_store.py        ChromaDB collection
  tools/function_tools.py    calculator, clock, KB search
  prod/backend.py            FastAPI app
  prod/llm_service.py        retries, rate limiting, model fallback
  prod/cache.py              thread-safe TTL cache
  prod/ui.py                 Streamlit chat UI
  assistant.py, main.py      single-pass assistant and CLI
eval/                        harness, query set, recorded results
docs/                        sample knowledge base
tests/                       offline unit tests
```

## Design notes

- **Why a ledger?** Raw tool outputs are noisy and expensive to replay. The
  agent keeps a capped list of compact notes (with source names and distance
  buckets) and re-reads that each step, which keeps prompts small and stable.
- **Why deterministic IDs?** Chunk IDs are a SHA-256 of the content and chunk
  index, so re-ingesting the same corpus upserts instead of duplicating.
- **Why a safe calculator?** The model's expressions are untrusted input. The
  calculator parses an AST allowlist instead of calling `eval`.
- **Why a confidence guardrail?** The model can answer from parametric memory
  when retrieval finds nothing. If the agent searched, found no relevant
  chunks, and still claimed high confidence, the answer is downgraded.

## License

MIT — see [`LICENSE`](LICENSE).
