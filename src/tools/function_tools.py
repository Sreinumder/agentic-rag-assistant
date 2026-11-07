from __future__ import annotations

import ast
import datetime
import math
import operator
from collections.abc import Callable
from typing import Any

# Expressions come from model output, so they are treated as untrusted input.
MAX_EXPRESSION_LENGTH = 500
MAX_NODES = 200
MAX_POWER = 1000

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses syntax outside the calculator's allowlist."""


class _Evaluator:
    def __init__(self) -> None:
        self.nodes = 0

    def evaluate(self, node: ast.AST) -> Any:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise UnsafeExpressionError("expression is too complex")

        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsafeExpressionError("only numeric literals are allowed")
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise UnsafeExpressionError(f"operator {op_type.__name__} is not allowed")
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if op_type is ast.Pow and not self._power_is_bounded(right):
                raise UnsafeExpressionError("exponent is too large")
            return _BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise UnsafeExpressionError(f"operator {op_type.__name__} is not allowed")
            return _UNARY_OPS[op_type](self.evaluate(node.operand))

        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise UnsafeExpressionError(f"unknown name {node.id!r}")

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.evaluate(item) for item in node.elts]

        if isinstance(node, ast.Call):
            func = self._resolve_function(node.func)
            args = [self.evaluate(arg) for arg in node.args]
            return func(*args)

        raise UnsafeExpressionError(f"syntax {type(node).__name__} is not allowed")

    @staticmethod
    def _power_is_bounded(exponent: Any) -> bool:
        return isinstance(exponent, (int, float)) and abs(exponent) <= MAX_POWER

    @staticmethod
    def _resolve_function(func: ast.AST) -> Callable[..., Any]:
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            # Accept ``math.sqrt`` in addition to ``sqrt``.
            if func.value.id != "math":
                raise UnsafeExpressionError(f"module {func.value.id!r} is not allowed")
            name = func.attr
        else:
            raise UnsafeExpressionError("only direct function calls are allowed")

        if name not in _FUNCTIONS:
            raise UnsafeExpressionError(f"function {name!r} is not allowed")
        return _FUNCTIONS[name]


def calculate_expression(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression without using ``eval``.

    Returns ``{"result": ..., "expression": ...}`` on success and
    ``{"error": ..., "expression": ...}`` on invalid or unsafe input.
    """
    if not isinstance(expression, str):
        return {"error": "expression must be a string", "expression": expression}

    expression = expression.strip()
    if not expression:
        return {"error": "empty expression", "expression": expression}
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"error": "expression is too long", "expression": expression}

    try:
        tree = ast.parse(expression, mode="eval")
        result = _Evaluator().evaluate(tree)
    except UnsafeExpressionError as exc:
        return {"error": str(exc), "expression": expression}
    except SyntaxError:
        return {"error": "invalid syntax", "expression": expression}
    except ZeroDivisionError:
        return {"error": "division by zero", "expression": expression}
    except (OverflowError, ValueError, TypeError) as exc:
        return {"error": str(exc), "expression": expression}

    if isinstance(result, complex):
        return {"error": "complex results are not supported", "expression": expression}
    return {"result": result, "expression": expression}


def get_current_datetime() -> dict[str, str]:
    now = datetime.datetime.now()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


def search_knowledge_base(query: str, vector_store: Any = None) -> dict[str, Any]:
    if vector_store is None:
        return {"error": "Vector store not initialized", "results": []}
    results = vector_store.query(query, top_k=3)
    return {"results": results, "query": query}


TOOL_DEFINITIONS = [
    {
        "name": "calculate_expression",
        "description": "Evaluate a mathematical expression safely. Supports basic arithmetic, sqrt, log, trig functions, and constants (pi, e).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(16) + pi'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time information including day of the week.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the document knowledge base for relevant information using semantic search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents.",
                }
            },
            "required": ["query"],
        },
    },
]

TOOL_MAP = {
    "calculate_expression": calculate_expression,
    "get_current_datetime": lambda: get_current_datetime(),
    "search_knowledge_base": search_knowledge_base,
}
