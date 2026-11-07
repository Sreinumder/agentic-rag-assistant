from __future__ import annotations

import json
from typing import Any

import pytest

from src.agents.research_agent import ResearchSession, _extract_json

FINALIZE = json.dumps(
    {
        "action": "finalize",
        "args": {
            "answer": "Supervised learning uses labelled examples.",
            "confidence": "high",
            "sources": ["docs/ml_basics.md"],
        },
        "reasoning": "the ledger answers the question",
    }
)

SEARCH = json.dumps(
    {
        "action": "search_knowledge_base",
        "args": {"query": "supervised learning", "top_k": 3},
        "reasoning": "need evidence",
    }
)

CLARIFY = json.dumps(
    {
        "action": "ask_clarification",
        "args": {"question": "What precision and recall should I use?"},
        "reasoning": "missing inputs",
    }
)


class ScriptedLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def generate_with_usage(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(prompt)
        text = self.outputs.pop(0) if self.outputs else FINALIZE
        return {
            "text": text,
            "usage": {
                "prompt_tokens": 10,
                "candidates_tokens": 5,
                "total_tokens": 15,
            },
        }


class FakeStore:
    def __init__(self, distance: float = 0.2):
        self.distance = distance
        self.queries: list[tuple[str, int]] = []

    def query(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self.queries.append((query, top_k or 0))
        return [
            {
                "content": "Supervised learning maps inputs to labelled outputs.",
                "metadata": {"source": "docs/ml_basics.md", "chunk_index": 0},
                "distance": self.distance,
            }
        ]


def make_session(
    outputs: list[str],
    store: FakeStore | None = None,
    **kwargs: Any,
) -> ResearchSession:
    return ResearchSession(
        "What is supervised learning?",
        vector_store=store or FakeStore(),
        llm=ScriptedLLM(outputs),
        max_steps=kwargs.pop("max_steps", 5),
        fail_injection=kwargs.pop("fail_injection", "none"),
        pace_seconds=0,
        **kwargs,
    )


def test_search_then_finalize() -> None:
    store = FakeStore(distance=0.2)
    session = make_session([SEARCH, FINALIZE], store)

    result = session.simulate()

    assert result["status"] == "finalized"
    assert result["steps"] == 2
    assert result["confidence"] == "high"
    assert result["evidence_relevant"] is True
    assert result["degraded"] is False
    assert result["sources"] == ["docs/ml_basics.md"]
    assert [call["action"] for call in result["tool_calls"]] == [
        "search_knowledge_base",
        "finalize",
    ]
    assert result["tokens"]["total_tokens"] == 30
    assert store.queries == [("supervised learning", 3)]


def test_out_of_kb_answer_is_downgraded() -> None:
    session = make_session([SEARCH, FINALIZE], FakeStore(distance=0.9))

    result = session.simulate()

    assert result["confidence"] == "low"
    assert result["evidence_relevant"] is False
    assert result["degraded"] is True
    assert "confidence_downgraded_unverified" in result["failures"]


def test_tool_failure_is_reported_and_downgrades_confidence() -> None:
    session = make_session(
        [SEARCH, FINALIZE],
        fail_injection="tool_unavailable",
    )

    result = session.simulate()

    assert result["status"] == "finalized"
    assert result["confidence"] == "low"
    assert result["degraded"] is True
    assert any("tool_error_search_knowledge_base" == f for f in result["failures"])


def test_malformed_retrieval_is_recoverable() -> None:
    session = make_session([SEARCH, FINALIZE], fail_injection="malformed_retrieval")

    result = session.simulate()

    assert result["status"] == "finalized"
    assert result["evidence_relevant"] is False
    assert "confidence_downgraded_unverified" in result["failures"]


def test_clarification_pauses_and_resumes() -> None:
    session = make_session([CLARIFY, FINALIZE])

    first = session.simulate()

    assert first["status"] == "clarification_pending"
    assert first["pending_question"] == "What precision and recall should I use?"
    assert session.pending_question is not None

    second = session.continue_with_reply("precision 0.6, recall 0.4")

    assert second["status"] == "finalized"
    assert second["clarified"] is True
    assert second["pending_question"] is None


def test_continue_without_pending_raises() -> None:
    session = make_session([FINALIZE])
    session.simulate()
    with pytest.raises(RuntimeError):
        session.continue_with_reply("nope")


def test_step_limit_produces_honest_fallback() -> None:
    session = make_session([SEARCH, SEARCH, SEARCH], max_steps=2)

    result = session.simulate()

    assert result["steps"] == 2
    assert result["confidence"] == "low"
    assert result["degraded"] is True
    assert "step_limit_reached" in result["failures"]
    assert "could not complete" in result["answer"]


def test_malformed_decision_retries_once() -> None:
    session = make_session(["not json at all", FINALIZE])

    result = session.simulate()

    assert result["status"] == "finalized"
    assert any(f.startswith("unparseable_decision") for f in result["failures"])
    assert any(t["action"] == "parse_retry" for t in result["tool_calls"])


def test_malformed_decision_twice_forces_fallback() -> None:
    session = make_session(["nope", "still nope"])

    result = session.simulate()

    assert result["status"] == "finalized"
    assert "forced_parse_failed" in result["failures"]
    assert result["confidence"] == "low"


def test_unknown_action_retries() -> None:
    bad = json.dumps({"action": "delete_everything", "args": {}})
    session = make_session([bad, FINALIZE])

    result = session.simulate()

    assert result["status"] == "finalized"
    assert any(f.startswith("unknown_action_") for f in result["failures"])


def test_calculator_note_is_added() -> None:
    calc = json.dumps(
        {
            "action": "calculate_expression",
            "args": {"expression": "0.6 * 0.4 / (0.6 + 0.4)"},
            "reasoning": "compute F1",
        }
    )
    session = make_session([calc, FINALIZE])

    result = session.simulate()

    assert result["status"] == "finalized"
    assert any(n["label"] == "calculate" for n in result["notes"])
    assert any("0.24" in n["detail"] for n in result["notes"])


def test_ledger_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import Config

    monkeypatch.setattr(Config, "EVIDENCE_MAX_NOTES", 2)
    session = make_session([SEARCH] * 5, max_steps=5)

    result = session.simulate()

    assert len(result["notes"]) <= 2
    assert "ledger_trimmed" in result["failures"]


@pytest.mark.parametrize(
    ("text", "expected_action"),
    [
        (FINALIZE, "finalize"),
        (f"```json\n{FINALIZE}\n```", "finalize"),
        (f"Here is my choice: {FINALIZE} Thanks.", "finalize"),
        ("{'action': 'finalize', 'args': {'answer': 'ok', 'confidence': 'low'}}", "finalize"),
        ("no json here", None),
    ],
)
def test_extract_json(text: str, expected_action: str | None) -> None:
    parsed = _extract_json(text)
    if expected_action is None:
        assert parsed is None
    else:
        assert parsed is not None
        assert parsed["action"] == expected_action


def test_invalid_fail_injection_rejected() -> None:
    with pytest.raises(ValueError):
        make_session([FINALIZE], fail_injection="nonsense")
