"""Evaluation harness for the agentic research loop (stdlib only).

Measures, per test query:
  - task completion (category-specific pass criteria)
  - tool-call correctness (valid action names, valid args, expected tools used)
  - trajectory length (steps) and whether it is reasonable for the query
  - token/cost accounting
  - failure log with the hard / soft / cascading-soft taxonomy
  - wall-clock time

Usage (from the project root):
    .venv/bin/python eval/harness.py [--report]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
QUERIES_FILE = Path(__file__).resolve().parent / "queries.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT_FILE = Path(__file__).resolve().parent / "RESULTS.md"

VALID_TOOL_ARGS = {
    "search_knowledge_base": ("query",),
    "calculate_expression": ("expression",),
    "get_current_datetime": (),
    "ask_clarification": ("question",),
    "finalize": ("answer", "confidence", "sources"),
}
VALID_CONFIDENCE = ("high", "medium", "low")

TOOL_NAMES = set(VALID_TOOL_ARGS)


def ensure_kb() -> Any:
    from src.rag.vector_store import VectorStore

    vs = VectorStore()
    if vs.count == 0:
        from src.assistant import AIAssistant

        AIAssistant().ingest_documents(str(BASE / "docs"))
    return vs


def run_case(qmeta: dict[str, Any], vector_store: Any, pace: float) -> dict[str, Any]:
    from src.agents.research_agent import ResearchSession

    session = ResearchSession(
        qmeta["query"],
        vector_store=vector_store,
        fail_injection=qmeta.get("fail_injection", "none"),
        pace_seconds=pace,
    )
    t0 = time.time()
    try:
        result = session.simulate({})
        clarifications = 0
        while result.get("status") == "clarification_pending" and result.get("pending_question"):
            clarifications += 1
            if clarifications > 3:
                break
            reply = qmeta.get("clarification_reply", "use precision 0.6 and recall 0.4")
            result = session.continue_with_reply(reply)
    except Exception as exc:  # noqa: BLE001 - the harness must capture hard failures
        result = {
            "query": qmeta["query"],
            "status": "error",
            "answer": "",
            "confidence": "",
            "sources": [],
            "steps": session._steps_taken,
            "evidence_relevant": False,
            "degraded": True,
            "tool_calls": session.transcript,
            "notes": session.notes,
            "failures": session.failures + [f"raised_{type(exc).__name__}"],
            "tokens": session.usage_total,
        }
    result["wall_seconds"] = round(time.time() - t0, 2)
    result["query_id"] = qmeta["id"]
    return result


# ---------------------------------------------------------------- correctness


def validate_tool_call(action: str, args: dict[str, Any]) -> tuple[bool, str]:
    if action not in TOOL_NAMES:
        return False, f"unknown action {action!r}"
    if isinstance(args, dict) and args.get("sources") is not None:
        if not isinstance(args["sources"], list) or not all(isinstance(s, str) for s in args["sources"]):
            return False, "finalize.sources must be a list of strings"
    if action == "finalize":
        if not str(args.get("answer", "")).strip():
            return False, "finalize.answer empty"
        if str(args.get("confidence", "")).lower() not in VALID_CONFIDENCE:
            return False, f"finalize.confidence invalid: {args.get('confidence')}"
        return True, ""
    if action == "search_knowledge_base":
        if not str(args.get("query", "")).strip():
            return False, "search.query empty"
        top_k = args.get("top_k", 3)
        if not (isinstance(top_k, int) and not isinstance(top_k, bool) and 1 <= top_k <= 6):
            return False, f"search.top_k invalid: {top_k!r}"
        return True, ""
    if action == "calculate_expression":
        if not str(args.get("expression", "")).strip():
            return False, "calculate.expression empty"
        return True, ""
    for key in VALID_TOOL_ARGS[action]:
        if action == "ask_clarification" and not str(args.get(key, "")).strip():
            return False, f"{action}.{key} empty"
    return True, ""


def tool_report(tool_calls: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    ok = True
    problems: list[str] = []
    for call in tool_calls:
        action = call["action"]
        if action in ("user_reply", "parse_retry"):
            continue
        valid, problem = validate_tool_call(action, call.get("args"))
        if not valid:
            ok = False
            problems.append(f"step{call['step']} {action}: {problem}")
    return ok, problems


# -------------------------------------------------------------------- judging


def judge(qmeta: dict[str, Any], r: dict[str, Any]) -> tuple[bool, list[str]]:
    cat = qmeta["category"]
    reasons: list[str] = []
    ok = True

    def fail(reason: str) -> None:
        nonlocal ok
        ok = False
        reasons.append(reason)

    tools = [t["action"] for t in r.get("tool_calls", [])]
    used = set(tools)
    for expected in qmeta.get("expected_tools", []):
        if expected not in used:
            fail(f"missing expected tool {expected}")

    if r.get("status") == "error" or not r.get("answer", "").strip():
        fail(f"status={r.get('status')}; answer empty")

    if cat == "kb_factual":
        if r.get("confidence") not in ("high", "medium"):
            fail(f"confident KB answer expected, got confidence={r.get('confidence')}")
        if not r.get("evidence_relevant"):
            fail("no RELEVANT evidence surfaced for a KB question")
        src = qmeta.get("source")
        if src and not any(src in s for s in r.get("sources", [])):
            fail(f"did not cite expected source {src}")

    elif cat == "multi_source":
        searches = sum(1 for t in tools if t == "search_knowledge_base")
        if searches < 2:
            fail(f"cross-source question should query multiple times, got {searches} search")
        if not r.get("evidence_relevant"):
            fail("no RELEVANT evidence surfaced")
        if r.get("confidence") not in ("high", "medium"):
            fail(f"confidence={r.get('confidence')} too low to answer from evidence")

    elif cat in ("kb_plus_math", "math", "multi_turn"):
        numeric = qmeta.get("numeric")
        if numeric and numeric not in str(r.get("answer", "")):
            fail(f"numeric answer {numeric} not found in answer")
        if "calculate_expression" not in used:
            fail("calculation tool was not used")
        if cat == "kb_plus_math" and qmeta.get("source"):
            if not any(qmeta["source"] in s for s in r.get("sources", [])):
                fail(f"did not cite expected source {qmeta['source']}")

        if cat == "multi_turn" and not r.get("clarified"):
            fail("multi-turn query completed without asking/using a clarification")

    elif cat == "out_of_kb":
        # Either the agent self-degrades (low confidence) or it explicitly says
        # the knowledge base does not cover the query.
        answers_hnestly = "knowledge base" in str(r.get("answer", "")).lower()
        if r.get("confidence") != "low" and not answers_hnestly:
            fail(
                f"out-of-KB answer should degrade to low confidence or state the KB "
                f"lacks the info; got confidence={r.get('confidence')}"
            )

    elif cat in ("injected_tool_failure", "injected_malformed_retrieval"):
        # The point of the injection: the agent must recognize the failure and
        # refuse to answer confidently from no evidence.
        if r.get("confidence") not in ("low",):
            fail(f"injected failure should degrade confidence, got {r.get('confidence')}")
        if r.get("evidence_relevant"):
            fail("weird: evidence flagged relevant under an injected retrieval failure")

    return ok, reasons


def classify_failure(qmeta: dict[str, Any], r: dict[str, Any]) -> str:
    """Map the result to the hard / soft / cascading-soft taxonomy."""
    if r.get("status") == "error" or not r.get("answer", "").strip():
        return "hard"
    cat = qmeta["category"]
    flags = r.get("failures", [])
    injected = cat.startswith("injected_")

    if injected:
        handled = r.get("confidence") == "low" and not r.get("evidence_relevant")
        if handled:
            return "none"  # injected fault contained: recovered by degrading honestly
        return "cascading" if any("tool_error" in f for f in flags) else "soft"

    soft = 0
    if r.get("degraded"):
        soft += 1
    if any("unparseable" in f or ("tool_error" in f and "injected" not in cat) for f in flags):
        soft += 1
    if r.get("status") == "finalized" and r.get("confidence") == "low" and cat != "out_of_kb":
        soft += 1
    if "step_limit_reached" in flags:
        soft += 1

    if soft >= 2:
        return "cascading"
    if soft == 1:
        return "soft"
    return "none"


# ------------------------------------------------------------------- reporting


def render_table(records: list[dict[str, Any]]) -> str:
    header = "| id | category | status | conf | steps | tools | tokens | pass | failure class |"
    sep = "|----|----------|--------|------|-------|-------|--------|------|---------------|"
    rows = []
    for rec in records:
        qid = rec["query_id"]
        cat = rec["category"]
        status = rec["status"]
        conf = rec["confidence"] or "-"
        steps = rec["steps"]
        tools = ",".join(
            t["action"] for t in rec["tool_calls"] if t["action"] not in ("user_reply", "parse_retry")
        )
        tokens = rec["tokens"]["total_tokens"]
        passed = "PASS" if rec["pass"] else "FAIL"
        fcls = rec["failure_class"]
        rows.append(f"| {qid} | {cat} | {status} | {conf} | {steps} | {tools} | {tokens} | {passed} | {fcls} |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic-loop evaluation harness")
    ap.add_argument("--pace", type=float, default=4.2, help="seconds to sleep between agent LLM calls")
    ap.add_argument("--report", action="store_true", help="write eval/RESULTS.md")
    ap.add_argument(
        "--report-from",
        type=str,
        default="",
        help="rebuild the report from a saved run directory (no LLM calls)",
    )
    args = ap.parse_args()

    if args.report_from:
        run_dir = Path(args.report_from)
        output = []
        for path in sorted(run_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            output.append(json.loads(path.read_text(encoding="utf-8")))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        report = build_report(output, summary)
        print(report)
        if args.report:
            REPORT_FILE.write_text(report, encoding="utf-8")
            print(f"[info] report written to {REPORT_FILE}")
        return

    cases = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    vs = ensure_kb()

    output: list[dict[str, Any]] = []
    for qmeta in cases:
        print(f"[info] running {qmeta['id']} ...", flush=True)
        rec = run_case(qmeta, vs, args.pace)
        valid, tool_problems = tool_report(rec.get("tool_calls", []))
        passed, reasons = judge(qmeta, rec)
        passed = passed and valid
        rec["tool_correct"] = valid
        rec["tool_problems"] = tool_problems
        rec["pass"] = passed
        rec["reasons"] = [
            *(reasons if reasons else ["ok"]),
            *(tool_problems if not valid else []),
        ]
        rec["category"] = qmeta["category"]
        rec["failure_class"] = classify_failure(qmeta, rec)
        output.append(rec)

    summary = {
        "total_queries": len(output),
        "passed": sum(1 for r in output if r["pass"]),
        "completion_rate": round(sum(1 for r in output if r["pass"]) / len(output), 3),
        "tool_correct": round(
            sum(1 for r in output if r["tool_correct"]) / len(output), 3
        ),
        "steps_avg": round(sum(r["steps"] for r in output) / len(output), 2),
        "steps_min": min(r["steps"] for r in output),
        "steps_max": max(r["steps"] for r in output),
        "tokens_total": sum(r["tokens"]["total_tokens"] for r in output),
        "tokens_prompt": sum(r["tokens"]["prompt_tokens"] for r in output),
        "tokens_candidates": sum(r["tokens"]["candidates_tokens"] for r in output),
        "cost_estimate_usd": round(sum(r.get("estimated_cost_usd", 0) for r in output), 4),
        "wall_seconds_total": round(sum(r.get("wall_seconds", 0) for r in output), 1),
        "failure_classes": {
            cls: sum(1 for r in output if r["failure_class"] == cls)
            for cls in ("hard", "soft", "cascading", "none")
        },
    }

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for rec in output:
        (run_dir / f"{rec['query_id']}.json").write_text(
            json.dumps(rec, indent=2, default=str), encoding="utf-8"
        )
    print("[info] results written to", run_dir)

    report = build_report(output, summary)
    print("\n" + report)
    if args.report:
        REPORT_FILE.write_text(report, encoding="utf-8")
        print(f"[info] report written to {REPORT_FILE}")


def build_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Results",
        "",
        f"_Automatically generated by `eval/harness.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}._",
        "",
        "## Summary",
        "",
        f"- Total test queries: **{summary['total_queries']}**",
        f"- Task completion rate: **{summary['completion_rate']:.0%}** ({summary['passed']}/{summary['total_queries']})",
        f"- Tool-call correctness: **{summary['tool_correct']:.0%}**",
        f"- Trajectory length: avg **{summary['steps_avg']}** steps (min {summary['steps_min']}, max {summary['steps_max']})",
        f"- Tokens: **{summary['tokens_total']}** total ({summary['tokens_prompt']} prompt / {summary['tokens_candidates']} completion)",
        f"- Estimated cost: ${summary['cost_estimate_usd']:.4f}",
        f"- Wall clock: {summary['wall_seconds_total']}s",
        f"- Failure classes: {json.dumps(summary['failure_classes'])}",
        "",
        "## Per-query results",
        "",
        render_table(records),
        "",
        "## Failure log",
        "",
    ]
    for rec in records:
        if rec["failure_class"] == "none" and rec["pass"]:
            continue
        lines.append(f"### {rec['query_id']} — {rec['failure_class']}")
        lines.append("")
        lines.append(f"- status: {rec['status']}, confidence: {rec['confidence'] or '-'}")
        lines.append(f"- reasons: {rec['reasons']}")
        lines.append(f"- failures: {rec['failures']}")
        if "confidence_downgraded_unverified" in rec["failures"]:
            lines.append(
                "- **recovered**: the model first proposed an ungrounded confident answer and "
                "the self-verification guardrail downgraded it to low confidence (expected "
                "behavior for out-of-KB queries — a *soft event avoided at the guardrail*)."
            )
        if rec["tool_problems"]:
            lines.append(f"- tool problems: {rec['tool_problems']}")
        lines.append(f"- steps: {rec['steps']}, tokens: {rec['tokens']['total_tokens']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()