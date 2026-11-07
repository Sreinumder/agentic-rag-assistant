from __future__ import annotations

import json
import math
import datetime
from typing import Any


def calculate_expression(expression: str) -> dict[str, Any]:
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


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
