from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Config
from src.llm.gemini_client import GeminiLLM
from src.rag.document_loader import load_documents, chunk_documents
from src.rag.vector_store import VectorStore
from src.tools.function_tools import TOOL_DEFINITIONS, TOOL_MAP


class AIAssistant:
    def __init__(self):
        Config.validate()
        self.vector_store = VectorStore()
        self.llm = GeminiLLM()
        self._tools_enabled = True
        self._tool_names = {t["name"] for t in TOOL_DEFINITIONS}

    def ingest_documents(self, source: str | Path) -> int:
        docs = load_documents(source)
        chunks = chunk_documents(docs)
        count = self.vector_store.add_documents(chunks)
        return count

    def chat(self, user_message: str) -> dict[str, Any]:
        context = self._retrieve_context(user_message)

        if self._tools_enabled:
            response = self._handle_tool_calls(user_message, context)
        else:
            response = self.llm.generate(user_message, context=context)

        return {
            "user_message": user_message,
            "response": response.get("text", ""),
            "tool_calls": response.get("tool_calls", []),
            "context_used": bool(context),
        }

    def chat_structured(self, user_message: str, schema: dict) -> dict[str, Any]:
        context = self._retrieve_context(user_message)
        response = self.llm.generate_structured(user_message, schema, context=context)
        return {
            "user_message": user_message,
            "response": response.get("text", ""),
            "structured_data": response.get("data"),
            "is_valid_json": response.get("structured", False),
            "context_used": bool(context),
        }

    def _retrieve_context(self, query: str) -> str:
        results = self.vector_store.query(query, top_k=Config.TOP_K)
        if not results:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {r['metadata'].get('source', 'unknown')}]\n{r['content']}"
            for r in results
        )

    def _handle_tool_calls(self, user_message: str, context: str) -> dict[str, Any]:
        llm_with_tools = GeminiLLM(tools=TOOL_DEFINITIONS)
        response = llm_with_tools.generate_with_tools(user_message, context=context)

        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            return response

        for call in tool_calls:
            func_name = call["name"]
            func_args = call["args"]

            if func_name == "search_knowledge_base":
                func_args["vector_store"] = self.vector_store

            tool_func = TOOL_MAP.get(func_name)
            if tool_func:
                result = tool_func(**func_args)
                follow_up = llm_with_tools.handle_tool_response(func_name, result)
                if follow_up.get("text"):
                    response["text"] = follow_up["text"]
                response["tool_calls"].append(
                    {"name": func_name, "args": func_args, "result": result}
                )
            else:
                response["tool_calls"].append(
                    {"name": func_name, "args": func_args, "error": f"Unknown tool: {func_name}"}
                )

        return response
