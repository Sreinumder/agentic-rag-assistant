# Architecture

This document describes how a request moves through the system, how the
agentic loop makes decisions, and where the resilience mechanisms live.

## Components

| Component | File | Responsibility |
| --- | --- | --- |
| CLI assistant | `src/assistant.py`, `src/main.py` | Single-pass chat, ingestion, structured output, `/research` command. |
| Research agent | `src/agents/research_agent.py` | Step loop, evidence ledger, tool dispatch, guardrails. |
| Agent protocol | `src/agents/research_skill.md` | Full action protocol, loaded into the system prompt on first tool use. |
| Gemini client | `src/llm/gemini_client.py` | Text/structured generation and token usage. |
| LLM service | `src/prod/llm_service.py` | Retry, rate limiting, model fallback, degraded responses. |
| Cache | `src/prod/cache.py` | Thread-safe TTL cache with size-bounded eviction. |
| Backend | `src/prod/backend.py` | FastAPI routes, request models, session store. |
| UI | `src/prod/ui.py` | Streamlit chat client for the backend. |
| Loader | `src/rag/document_loader.py` | Walks files, chunks text, attaches source metadata. |
| Embeddings | `src/rag/embeddings.py` | Gemini embedding calls. |
| Vector store | `src/rag/vector_store.py` | ChromaDB collection with deterministic IDs and upserts. |
| Tools | `src/tools/function_tools.py` | AST calculator, clock, knowledge-base search. |
| Evaluation | `eval/harness.py` | Scenario runner and report generator. |

## Request flow: `/research`

```
client
  │  POST /research {message}
  ▼
backend.research()
  │  no active session? → ResearchSession(message, vector_store)
  ▼
ResearchSession.run()
  │
  ├─ step 1 ─► prompt: task + empty ledger + step budget
  │             LLM returns {"action": "search_knowledge_base", ...}
  │             dispatch → VectorStore.query()
  │             append compact note: q, RELEVANT/MARGINAL counts,
  │             best distance, best excerpt, sources
  │
  ├─ step 2 ─► prompt: task + ledger
  │             LLM returns {"action": "finalize", "args": {...}}
  │             guardrail may downgrade confidence
  │
  ▼
{answer, confidence, sources, steps, degraded, tool_calls, tokens}
```

The backend keeps sessions in a process-local dictionary keyed by the original
message. Sending `reply` on a paused session calls
`session.continue_with_reply(reply)` and resumes the same ledger.

## The decision protocol

Every model turn must return one JSON object:

```json
{"action": "<name>", "args": {...}, "reasoning": "<one line>"}
```

Allowed actions:

| Action | Arguments | Effect |
| --- | --- | --- |
| `search_knowledge_base` | `query`, optional `top_k` (1–6) | Queries ChromaDB; records distance buckets and an excerpt. |
| `calculate_expression` | `expression` | Evaluates arithmetic through an AST allowlist. |
| `get_current_datetime` | — | Returns local date and time. |
| `ask_clarification` | `question` | Pauses the loop; at most two clarifications are encouraged. |
| `finalize` | `answer`, `confidence`, `sources` | Ends the loop and returns the answer. |

Parsing tolerates markdown fences, surrounding prose, and Python-style
single-quoted dicts. A malformed decision gets one retry; after that the loop
ends with an explicit "could not complete" answer and a logged failure reason.

## Evidence ledger

Tool results are not replayed verbatim. Each result becomes a short note:

```
[search] q="why chunk overlap" | RELEVANT x2 / MARGINAL x1
         | best d=0.318 (docs/rag_guide.md) | excerpt: ...
[calc]   0.6 * 0.4 / (0.6 + 0.4) = 0.24
[error]  search_knowledge_base failed: ...
```

Notes are capped by `EVIDENCE_MAX_NOTES` and each note by
`EVIDENCE_NOTE_MAX_CHARS`. When the ledger overflows, the oldest notes are
dropped and `ledger_trimmed` is recorded. This keeps the prompt bounded while
preserving the most recent evidence.

Retrieved chunks are deduplicated by source within a search and sorted by
distance. Distances below `RETRIEVAL_RELEVANT_MAX` count as relevant; distances
up to `RETRIEVAL_MARGINAL_MAX` count as marginal. Only the best chunk's excerpt
is included, keeping the note small.

## Stopping conditions

The loop stops when:

1. the model calls `finalize` with a non-empty answer;
2. the model calls `ask_clarification`, in which case control returns to the
   caller with `status = "clarification_pending"`;
3. processing a clarification reply pushes the step count to `MAX_STEPS`;
4. the decision cannot be parsed after one retry.

Every unfinished path produces a low-confidence fallback answer and appends a
failure reason such as `step_limit_reached` or
`forced_parse_failed`. The result is never silently empty.

## Guardrails

- **Confidence downgrade** — if the agent searched but never surfaced a chunk
  below `RETRIEVAL_RELEVANT_MAX`, a `high` or `medium` finalize is downgraded to
  `low` and `confidence_downgraded_unverified` is recorded. This catches
  out-of-KB questions answered from model memory.
- **Degraded flag** — a result is `degraded` when the step limit was hit or a
  retrieval-driven query ended without relevant evidence. Pure math and clock
  queries are not marked degraded.
- **Tool errors are recoverable** — a failed search appends an `[error]` note
  and returns control to the model instead of raising, so the model can retry
  or finalize honestly.
- **Safe evaluation** — expressions are parsed with `ast` and evaluated only
  for numeric literals, approved unary/binary operators, and an allowlist of
  functions and constants. Names, attributes, calls outside the allowlist, and
  oversized expressions are rejected.

## Resilience in the backend

`LLMService.generate` wraps each request in a retry loop:

1. enforce a sliding one-minute request budget (`RATE_LIMIT_RPM`);
2. call the primary model;
3. on failure, call the fallback model;
4. on repeated failure, raise `AllProvidersFailedError`, which the route maps
   to HTTP 503; rate-limit exhaustion maps to HTTP 429.

`ResearchSession._call_llm` additionally honors Gemini's `retry in Ns` hint on
429/503 responses with a capped backoff.

The TTL cache is keyed by a SHA-256 of prompt, context, and temperature. It is
protected by a re-entrant lock because FastAPI serves requests from a thread
pool, and it evicts the oldest quarter when full.

## Ingestion

```
source path ─► load_documents() ─► chunk_documents() ─► add_documents()
                file walk            recursive split          embed + upsert
```

- Supported extensions: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log`, `.yaml`,
  `.yml`, `.toml`.
- Each chunk keeps source path and chunk index metadata.
- IDs are `sha256(source + content + chunk_index)`, so re-ingesting an unchanged
  corpus updates existing rows instead of creating duplicates.
- Embeddings are written in batches of 100.

## Evaluation harness

`eval/harness.py` runs each query in `eval/queries.json` and checks:

- completion status and answer presence;
- tool-call validity (action names, required arguments, `top_k` range);
- expected tools used;
- step bounds per scenario;
- numeric answers when a scenario defines one;
- token, cost, and wall-clock totals.

Failure injection modes (`tool_unavailable`, `malformed_retrieval`,
`max_steps`) verify that the loop degrades gracefully instead of crashing.
Results are written to `eval/results/<run>/` and summarized in
`eval/RESULTS.md`.
