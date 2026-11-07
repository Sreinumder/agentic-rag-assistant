# Research Agent Skill — full protocol

This protocol is loaded into the model context only after the model chooses to
use a tool for the first time (progressive disclosure). Until then the model
works from the concise action list in its base instructions.

## Respond with a single JSON object

```json
{ "action": "...", "args": { ... }, "reasoning": "one line explaining this step" }
```

## Available actions

| action | args | effect |
|---|---|---|
| `search_knowledge_base` | `{"query": str, "top_k": int (1-6)}` | Semantic search over the knowledge base. Returns scored evidence notes. |
| `calculate_expression` | `{"expression": str}` | Safe math evaluator (sqrt, log, trig, pi, e). |
| `get_current_datetime` | `{}` | Current date/time. |
| `ask_clarification` | `{"question": str}` | Stop the loop and ask the user for missing information. |
| `finalize` | `{"answer": str, "confidence": "high"\|"medium"\|"low", "sources": [str]}` | End the loop and answer. |

## Strategy

1. `search_knowledge_base` first. Evaluate the returned evidence notes, which
   tag each hit as `RELEVANT`, `MARGINAL`, or `IRRELEVANT` based on semantic
   distance.
2. Do **not** search a second time for the same facts. If the ledger already has
   `RELEVANT` evidence that answers the question, `finalize` immediately. A
   comprehensive answer is not required; answer only what the documents support.
3. Search again only when there is a specific, new gap: a sub-part of the
   question with no `RELEVANT` support, or a claim that must be cross-checked
   across a different document. If a repeat search returns the same sources as
   an earlier one, finalize with what you have.
4. If part of the answer is computable (e.g. the F1 score from precision and
   recall), compute it with `calculate_expression` instead of guessing.
5. If the query is ambiguous or is missing information only the user can
   provide, call `ask_clarification`. Your question must be specific.
6. **Error recovery.** If a tool returns an error (the evidence note says
   `[error]` or the last action status is `error`), you gained no evidence from
   that call. Retry **once** with a fresh query. If the same error repeats, the
   service is down: stop searching, `finalize` with `confidence: "low"`, and
   explicitly state that the relevant sources/retrieval were unavailable.
   **Never** report `high` confidence from a failed tool call, and never treat
   a failed search as evidence.
7. Only call `finalize` when you can answer from the evidence ledger with an
   honest confidence value, and list the actual source filenames from the
   ledger. Never fabricate sources.
8. If you have searched 3+ times and found no `RELEVANT` evidence, finalize
   with `confidence: "low"` and clearly state the knowledge base does not
   cover the topic.