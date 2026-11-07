from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import Config
from src.tools.function_tools import calculate_expression, get_current_datetime

if TYPE_CHECKING:
    from src.llm.gemini_client import GeminiLLM
    from src.rag.vector_store import VectorStore

SKILL_FILE = Path(__file__).parent / "research_skill.md"

BASE_INSTRUCTIONS_HEAD = (
    "You are a research agent that answers a user query by gathering evidence from a "
    "document knowledge base and deciding, step by step, what to do next.\n\n"
    "You respond to each step with a single JSON object:\n"
    '{"action": "<name>", "args": {...}, "reasoning": "<one line>"}\n\n'
    "Available actions (one line each):\n"
    "- search_knowledge_base: search the knowledge base.\n"
    "- calculate_expression: evaluate a math expression.\n"
    "- get_current_datetime: current date/time.\n"
    "- ask_clarification: stop and ask the user a question.\n"
    "- finalize: end and answer.\n\n"
    "Core rule: once the ledger contains RELEVANT evidence that answers the "
    "question, call finalize. Do not re-search the same facts.\n\n"
    "Full protocol is provided once you use a tool for the first time."
)

MAX_CLARIFICATIONS = 2
MAX_PARSE_RETRIES = 1

ALLOWED_ACTIONS = {
    "finalize",
    "search_knowledge_base",
    "calculate_expression",
    "get_current_datetime",
    "ask_clarification",
}


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse a decision dict from model output.

    Tolerates markdown fences, prose around the JSON, and single-quoted
    (Python-style) dicts that Gemini sometimes emits.
    """
    candidates: list[str] = []
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[m.start() : j + 1])
                    break

    for cand in candidates:
        cand = cand.replace("```json", "").replace("```", "")
        for loader in (json.loads, _literal_load):
            try:
                parsed = loader(cand)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _literal_load(s: str) -> Any:
    import ast

    obj = ast.literal_eval(s)
    return _jsonify(obj)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    return obj


class ResearchSession:
    """Single-agent loop: the model decides each next action from an evidence ledger.

    The loop is not a fixed pipeline wrapped in a loop: the model chooses between
    searching again (with a new query it crafts), computing, or finalizing based on
    the accumulated evidence. Stopping conditions: `finalize`, `ask_clarification`
    (multi-turn), or the hard `max_steps` limit.
    """

    def __init__(
        self,
        query: str,
        vector_store: VectorStore | None = None,
        max_steps: int | None = None,
        fail_injection: str | None = None,
        llm: GeminiLLM | None = None,
        model: str | None = None,
        pace_seconds: float | None = None,
    ):
        self.query = query
        self.vector_store = vector_store if vector_store is not None else self._make_store()
        self.llm = llm if llm is not None else self._make_llm()
        self.model = model or (Config.AGENT_MODEL or Config.GEMINI_MODEL)
        self.pace_seconds = (
            Config.AGENT_PACE_SECONDS if pace_seconds is None else pace_seconds
        )
        self.fail_injection = (
            fail_injection if fail_injection is not None else Config.FAILURE_INJECTION
        )
        if self.fail_injection not in (
            "none",
            "tool_unavailable",
            "malformed_retrieval",
            "max_steps",
        ):
            raise ValueError(f"Unknown failure injection mode: {self.fail_injection}")

        # The injected "max_steps" failure forces a 1-step budget regardless of
        # any explicitly passed value so the loop always exhausts its stopping
        # condition during the failure-injection test.
        if self.fail_injection == "max_steps":
            self.max_steps = 1
        else:
            self.max_steps = max(1, min(max_steps or Config.MAX_STEPS, Config.MAX_STEPS))

        self.notes: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.usage_total = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
        self._steps_taken = 0
        self._skill_activated = False
        self._parse_retries = 0
        self._clarification_count = 0
        self._pending_question: str | None = None
        self._provided_reply: str | None = None
        self._awaiting_reply = False
        self._did_clarify = False
        self._finalized: dict[str, Any] | None = None

    # --------------------------------------------------------------- factories

    @staticmethod
    def _make_store() -> VectorStore:
        # Deferred import keeps the agent testable with fakes and avoids loading
        # ChromaDB/Google SDKs when they are not needed.
        from src.rag.vector_store import VectorStore as _VectorStore

        return _VectorStore()

    @staticmethod
    def _make_llm() -> GeminiLLM:
        from src.llm.gemini_client import GeminiLLM as _GeminiLLM

        return _GeminiLLM()

    # ------------------------------------------------------------- simulation

    def simulate(self, replies: dict[str, str] | None = None) -> dict[str, Any]:
        """Run with automatic canned replies for any clarifications (harness use)."""
        while self._steps_taken < self.max_steps:
            if self._awaiting_reply:
                if replies is None or self._pending_question not in replies:
                    break
                self.continue_with_reply(replies[self._pending_question])
                continue
            outcome = self._step()
            if outcome == "finalized":
                break
            if outcome == "forced_parse_failed":
                break
            if outcome == "clarification" and replies is None:
                break
        if self._finalized is None and not self._awaiting_reply:
            self._force_unfinished()
        return self.to_result()

    # -------------------------------------------------------------- public loop

    def run(self) -> dict[str, Any]:
        return self.simulate()

    def continue_with_reply(self, reply: str) -> dict[str, Any]:
        if not self._awaiting_reply or self._pending_question is None:
            raise RuntimeError("No pending clarification to answer")
        self._provided_reply = reply
        self._awaiting_reply = False
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": "user_reply",
                "args": {"reply": reply[:400]},
                "status": "ok",
                "output_summary": f"user answered: {reply[:120]}",
            }
        )
        self._steps_taken += 1
        self._did_clarify = True
        return self.simulate()

    @property
    def is_finalized(self) -> bool:
        """True once the loop has finalized (safe to query the result)."""
        return self._finalized is not None

    @property
    def pending_question(self) -> str | None:
        """The clarification question the agent is currently waiting for, if any."""
        return self._pending_question if self._awaiting_reply else None

    # ------------------------------------------------------------------ internal

    def _step(self) -> str:
        self._steps_taken += 1
        prompt = self._build_step_prompt()
        gen = self._call_llm(prompt, self._system_instruction())
        self._add_usage(gen.get("usage", {}))
        decision = _extract_json(gen.get("text", ""))

        if decision is None:
            self.failures.append(f"unparseable_decision_step_{self._steps_taken}")
            if self._parse_retries < MAX_PARSE_RETRIES:
                self._parse_retries += 1
                return self._retry_after_parse_error(gen.get("text", ""))
            self.failures.append("forced_parse_failed")
            self._force_unfinished()
            return "forced_parse_failed"

        return self._dispatch(decision)

    def _call_llm(self, prompt: str, system_instruction: str) -> dict[str, Any]:
        """Wrap the LLM call with pacing and quota-aware retries.

        The free-tier Gemini API returns 429/503 with a `retry in Ns` hint when the
        per-minute request quota is exhausted; we honor it instead of failing.
        """
        if self.pace_seconds > 0:
            time.sleep(self.pace_seconds)
        for attempt in range(1, 4):
            try:
                return self.llm.generate_with_usage(
                    prompt, system_instruction=system_instruction, model=self.model
                )
            except Exception as e:  # noqa: BLE001 - we carefully re-raise below
                msg = str(e)
                if re.search(r"429|503|RESOURCE_EXHAUSTED|UNAVAILABLE", msg):
                    match = re.search(r"retry in ([0-9.]+)s", msg, re.IGNORECASE)
                    delay = min(float(match.group(1)) + 0.5 if match else 5.0 * attempt, 60.0)
                    self.failures.append(f"llm_rate_limit_backoff_{attempt}")
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("LLM call failed after 3 attempts")

    def _dispatch(self, decision: dict[str, Any]) -> str:
        action = decision.get("action")
        args = decision.get("args") if isinstance(decision.get("args"), dict) else {}

        if action not in ALLOWED_ACTIONS:
            self.failures.append(f"unknown_action_{action}")
            return self._retry_after_parse_error(
                f"Action '{action}' is not allowed. Choose one of: {sorted(ALLOWED_ACTIONS)}"
            )

        if action == "finalize":
            answer = str(args.get("answer", ""))
            confidence = str(args.get("confidence", "low")).lower()
            if confidence not in ("high", "medium", "low"):
                confidence = "low"
            sources = args.get("sources")
            if not isinstance(sources, list):
                sources = []
            if not answer.strip():
                self._force_unfinished()
                return "forced_parse_failed"
            self._finalized = {
                "answer": answer,
                "confidence": confidence,
                "sources": [str(s) for s in sources if isinstance(s, str)],
            }
            self._apply_confidence_guardrail()
            confidence = self._finalized["confidence"]
            self.transcript.append(
                {
                    "step": self._steps_taken,
                    "action": action,
                    "args": {
                        "answer": answer,
                        "confidence": confidence,
                        "sources": sources,
                    },
                    "status": "final",
                    "output_summary": f"finalize ({confidence}): {answer[:120]}",
                }
            )
            return "finalized"

        if action == "ask_clarification":
            question = str(args.get("question", "")).strip()
            if not question:
                self._force_unfinished()
                return "forced_parse_failed"
            self._clarification_count += 1
            self._pending_question = question
            self._awaiting_reply = True
            self.transcript.append(
                {
                    "step": self._steps_taken,
                    "action": action,
                    "args": {"question": question},
                    "status": "paused",
                    "output_summary": question[:200],
                }
            )
            if self._clarification_count >= MAX_CLARIFICATIONS:
                self.failures.append("clarification_limit_reached")
            return "clarification"

        if action == "search_knowledge_base":
            self._skill_activated = True
            return self._exec_search(args)
        if action == "calculate_expression":
            self._skill_activated = True
            return self._exec_calculate(args)
        if action == "get_current_datetime":
            self._skill_activated = True
            return self._exec_datetime(args)
        return "error"

    def _exec_search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        try:
            top_k = max(1, min(int(args.get("top_k", 3)), 6))
        except (TypeError, ValueError):
            top_k = 3
        if not query:
            return self._tool_error(
                "search_knowledge_base", args, "empty query supplied by the model"
            )

        if self.fail_injection == "tool_unavailable":
            return self._tool_error(
                "search_knowledge_base",
                args,
                "tool unavailable (simulated outage): the knowledge base service is down",
            )

        hits = self.vector_store.query(query, top_k=top_k)
        if self.fail_injection == "malformed_retrieval":
            hits = []
            payload = {"error": "malformed_retrieval_payload", "results": []}
            return self._record_search_note(
                query, top_k, payload, note_text=(
                    f"search failed: malformed retrieval payload returned by the store "
                    f"(no usable results). Treat as no evidence."
                )
            )

        capped = hits[: Config.RETRIEVAL_HIT_CAP]
        seen = set()
        ranked: list[dict[str, Any]] = []
        for h in capped:
            src = h.get("metadata", {}).get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)
            ranked.append(h)
        ranked.sort(key=lambda h: float(h.get("distance", 9)))

        payload = {"query": query, "results": ranked}
        note = self._build_search_note(query, ranked)
        return self._record_search_note(query, top_k, payload, note_text=note)

    def _exec_calculate(self, args: dict[str, Any]) -> str:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return self._tool_error(calculate_expression.__name__, args, "empty expression")
        result = calculate_expression(expr)
        note = f"[calc] {expr} = {result.get('result', result)}"
        self._append_note("calculate", note, sources=[], raw=result)
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": "calculate_expression",
                "args": args,
                "status": "ok",
                "output_summary": note,
            }
        )
        return "ok"

    def _exec_datetime(self, args: dict[str, Any]) -> str:
        result = get_current_datetime()
        note = f"[time] {result.get('date')} {result.get('time')} ({result.get('day_of_week')})"
        self._append_note("datetime", note, sources=[], raw=result)
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": "get_current_datetime",
                "args": args,
                "status": "ok",
                "output_summary": note,
            }
        )
        return "ok"

    # ------------------------------------------------------------- ledger utils

    def _build_search_note(self, query: str, ranked: list[dict[str, Any]]) -> str:
        rel = [h for h in ranked if float(h.get("distance", 9)) <= Config.RETRIEVAL_RELEVANT_MAX]
        marg = [h for h in ranked if Config.RETRIEVAL_RELEVANT_MAX < float(h.get("distance", 9)) <= Config.RETRIEVAL_MARGINAL_MAX]
        best = ranked[0] if ranked else None
        budget = max(120, Config.EVIDENCE_NOTE_MAX_CHARS - 160)
        if best:
            src = best.get("metadata", {}).get("source", "unknown")
            excerpt = best.get("content", "")[:budget].replace("\n", " ")
            return (
                f"[search] q=\"{query}\" | RELEVANT x{len(rel)} / MARGINAL x{len(marg)} "
                f"| best d={float(best.get('distance', 9)):.3f} ({src}) | "
                f"excerpt: {excerpt}"
            )
        return f"[search] q=\"{query}\" | no hits"

    def _record_search_note(self, query, top_k, payload, note_text: str) -> str:
        sources = [
            h.get("metadata", {}).get("source", "unknown")
            for h in payload.get("results", [])
        ]
        self._append_note("search", note_text, sources=sources, raw=payload)
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": "search_knowledge_base",
                "args": {"query": query, "top_k": top_k},
                "status": "ok",
                "output_summary": note_text[:200],
            }
        )
        return "ok"

    def _tool_error(self, name: str, args: dict[str, Any], message: str) -> str:
        self.failures.append(f"tool_error_{name}")
        note = f"[error] {name} failed: {message}"
        self._append_note("error", note, sources=[], raw=None)
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": name,
                "args": args,
                "status": "error",
                "output_summary": note,
            }
        )
        # Return "ok" so the loop keeps iterating: the error note is fed back to
        # the model, which must decide how to recover instead of dying.
        return "ok"

    def _append_note(self, label: str, detail: str, sources: list[str], raw: Any) -> None:
        budget = Config.EVIDENCE_NOTE_MAX_CHARS
        detail = detail[:budget]
        self.notes.append({"step": self._steps_taken, "label": label, "detail": detail, "sources": sources})
        cap = Config.EVIDENCE_MAX_NOTES
        if len(self.notes) > cap:
            self.notes = self.notes[-cap:]
            self.failures.append("ledger_trimmed")

    # ------------------------------------------------------------- prompt build

    def _system_instruction(self) -> str:
        instr = BASE_INSTRUCTIONS_HEAD
        if self._skill_activated:
            instr += "\n\n" + SKILL_FILE.read_text(encoding="utf-8")
        return instr

    def _build_step_prompt(self) -> str:
        lines = [f"Current task: {self.query}", ""]
        if self._provided_reply is not None:
            lines.append(
                f"Clarification you asked and the user's answer:\n"
                f"Q: {self._pending_question}\nA: {self._provided_reply}\n"
            )
        if self.notes:
            ledger = "\n".join(f"- {n['detail']}" for n in self.notes)
            lines.append("Evidence ledger (compacted notes, raw outputs are not retained):\n" + ledger)
        else:
            lines.append("Evidence ledger: (empty)")
        last = self.transcript[-1] if self.transcript else None
        if last:
            lines.append(
                f"\nLast action: {last['action']} {last.get('output_summary', '')}"
            )
        lines.append(
            f"\nThis is step {self._steps_taken} of at most {self.max_steps}. "
            "Choose the next action. If the ledger already answers the question, "
            "set action=\"finalize\" now and do not search again for the same facts. "
            "Only finalize with an honest confidence value. Do not fabricate sources."
        )
        return "\n".join(lines)

    def _retry_after_parse_error(self, hint: str) -> str:
        self.transcript.append(
            {
                "step": self._steps_taken,
                "action": "parse_retry",
                "args": {},
                "status": "error",
                "output_summary": f"decision was not valid JSON; retrying. model said: {hint[:200]}",
            }
        )
        return self._step()

    def _apply_confidence_guardrail(self) -> None:
        """Self-verification guardrail: after a finalize decision, downgrade
        confidence to `low` when the agent searched but never surfaced RELEVANT
        evidence. This prevents confident answers that are not grounded in the
        knowledge base even if the model is tempted to answer from parametric
        memory (e.g. an out-of-KB query)."""
        if self._finalized is None:
            return
        searched = any(
            t.get("action") == "search_knowledge_base" for t in self.transcript
        )
        if not searched:
            return
        has_relevant = any(
            "RELEVANT x" in n["detail"] and "RELEVANT x0" not in n["detail"]
            for n in self.notes
            if n["label"] == "search"
        )
        if not has_relevant and self._finalized["confidence"] != "low":
            self._finalized["confidence"] = "low"
            self.failures.append("confidence_downgraded_unverified")

    def _force_unfinished(self) -> None:
        self._finalized = {
            "answer": (
                f"I could not complete this task within {self.max_steps} steps. "
                "The evidence I gathered is insufficient to give a confident answer."
            ),
            "confidence": "low",
            "sources": [],
        }
        self.failures.append("step_limit_reached")

    def _add_usage(self, usage: dict[str, Any]) -> None:
        for k in ("prompt_tokens", "candidates_tokens", "total_tokens"):
            self.usage_total[k] += int(usage.get(k, 0) or 0)

    def to_result(self) -> dict[str, Any]:
        searches = [n for n in self.notes if n["label"] == "search"]
        evidence_ok = any(
            "RELEVANT x" in n["detail"] and "RELEVANT x0" not in n["detail"]
            for n in searches
        )
        attempted_search = any(
            t.get("action") == "search_knowledge_base" for t in self.transcript
        )
        # A query that involves knowledge retrieval is "degraded" when the loop
        # gave up or never surfaced relevant evidence (would produce a
        # fabricatable/unverifiable answer). Pure computation/clock queries are
        # not degraded even without KB evidence.
        retrieval_degraded = attempted_search and not evidence_ok
        degraded = "step_limit_reached" in self.failures or retrieval_degraded
        total = self.usage_total
        cost = (
            total["prompt_tokens"] / 1000 * Config.USD_PER_1K_INPUT
            + total["candidates_tokens"] / 1000 * Config.USD_PER_1K_OUTPUT
        )
        return {
            "query": self.query,
            "status": (
                "finalized" if self._finalized is not None else "clarification_pending"
            ),
            "clarified": self._did_clarify,
            "pending_question": self._pending_question if self._awaiting_reply else None,
            "answer": (self._finalized or {}).get("answer", ""),
            "confidence": (self._finalized or {}).get("confidence", ""),
            "sources": (self._finalized or {}).get("sources", []),
            "steps": self._steps_taken,
            "max_steps": self.max_steps,
            "evidence_relevant": evidence_ok,
            "degraded": degraded,
            "tool_calls": self.transcript,
            "notes": self.notes,
            "failures": self.failures,
            "tokens": total,
            "estimated_cost_usd": round(cost, 6),
        }
