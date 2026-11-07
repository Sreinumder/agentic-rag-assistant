from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from src.config import Config


SYSTEM_PROMPT = Config.SYSTEM_PROMPT


class GeminiLLM:
    def __init__(self, tools: list[dict[str, Any]] | None = None):
        self._client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self._tools = tools
        self._history: list[types.Content] = []

    def generate(self, prompt: str, context: str = "") -> dict[str, Any]:
        full_prompt = self._build_prompt(prompt, context)
        response = self._client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            ),
        )
        self._history.append(types.Content(role="user", parts=[types.Part(text=full_prompt)]))
        return self._parse_response(response)

    def generate_structured(self, prompt: str, schema: dict, context: str = "") -> dict[str, Any]:
        full_prompt = self._build_structured_prompt(prompt, schema, context)
        response = self._client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                max_output_tokens=Config.MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
        parsed = self._parse_response(response)
        try:
            content = parsed.get("text", "")
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0]
            parsed_json = json.loads(json_match.strip())
            return {"structured": True, "data": parsed_json, "text": content}
        except (json.JSONDecodeError, IndexError):
            return {"structured": False, "data": None, "text": parsed.get("text", "")}

    def generate_with_tools(self, prompt: str, context: str = "") -> dict[str, Any]:
        full_prompt = self._build_prompt(prompt, context)
        chat = self._client.chats.create(
            model=Config.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            ),
        )
        response = chat.send_message(full_prompt)
        return self._parse_response(response)

    def generate_with_usage(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Single generation call that also returns Gemini token usage metadata.

        Used by the agentic loop for decision turns so token/cost accounting
        can be recorded per query.
        """
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction or SYSTEM_PROMPT,
            temperature=Config.TEMPERATURE if temperature is None else temperature,
            top_p=Config.TOP_P,
            max_output_tokens=max_output_tokens or Config.MAX_OUTPUT_TOKENS,
        )
        response = self._client.models.generate_content(
            model=model or (Config.AGENT_MODEL or Config.GEMINI_MODEL),
            contents=prompt,
            config=gen_config,
        )
        usage = getattr(response, "usage_metadata", None)
        return {
            "text": response.text if response.text else "",
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "candidates_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
            },
        }

    def _build_prompt(self, prompt: str, context: str) -> str:
        parts = []
        if context:
            parts.append(f"Context from knowledge base:\n---\n{context}\n---\n")
        parts.append(f"User query: {prompt}")
        return "\n\n".join(parts)

    def _build_structured_prompt(self, prompt: str, schema: dict, context: str) -> str:
        schema_str = json.dumps(schema, indent=2)
        parts = []
        if context:
            parts.append(f"Context from knowledge base:\n---\n{context}\n---\n")
        parts.append(f"User query: {prompt}")
        parts.append(f"\nPlease respond with valid JSON matching this schema:\n{schema_str}")
        return "\n\n".join(parts)

    def _parse_response(self, response) -> dict[str, Any]:
        result: dict[str, Any] = {"text": "", "tool_calls": []}
        if response.text:
            result["text"] = response.text
        return result

    def handle_tool_response(self, tool_name: str, tool_result: Any) -> dict[str, Any]:
        return {"text": json.dumps(tool_result), "tool_calls": []}
