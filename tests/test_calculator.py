from __future__ import annotations

import math

import pytest

from src.tools.function_tools import calculate_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-5 + 2", -3),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("2 ** 10", 1024),
        ("abs(-4)", 4),
        ("min(3, 1, 2)", 1),
        ("max([3, 1, 2])", 3),
        ("sum([1, 2, 3])", 6),
        ("round(3.14159, 2)", 3.14),
        ("pow(2, 8)", 256),
    ],
)
def test_arithmetic_and_functions(expression: str, expected: object) -> None:
    assert calculate_expression(expression)["result"] == expected


def test_constants_and_math_prefix() -> None:
    assert calculate_expression("sqrt(16) + pi")["result"] == pytest.approx(4 + math.pi)
    assert calculate_expression("math.sqrt(16)")["result"] == 4.0
    assert calculate_expression("cos(0)")["result"] == 1.0
    assert calculate_expression("log(e)")["result"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "lambda: 1",
        "x + 1",
        "'a' + 'b'",
        "True + 1",
        "math.sin",
        "math.atan2(1, 1)",
        "2 ** 100000",
        "1 / 0",
        "1 +",
        "(",
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    result = calculate_expression(expression)
    assert "error" in result
    assert "result" not in result


def test_rejects_oversized_expression() -> None:
    expression = " + ".join(["1"] * 1000)
    assert calculate_expression(expression)["error"] == "expression is too long"


def test_empty_and_non_string_input() -> None:
    assert calculate_expression("   ")["error"] == "empty expression"
    assert calculate_expression(None)["error"] == "expression must be a string"  # type: ignore[arg-type]


def test_expression_is_echoed_back() -> None:
    assert calculate_expression(" 1 + 1 ")["expression"] == "1 + 1"
