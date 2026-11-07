from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from src.config import Config

logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    pass


class AllProvidersFailedError(Exception):
    pass


class LLMService:
    """Production LLM client with retry, rate limiting, provider fallback, and graceful degradation."""

    def __init__(self):
        self._max_retries = int(Config.MAX_RETRIES)
        self._retry_delay = float(Config.RETRY_DELAY)
        self._max_requests_per_minute = int(Config.RATE_LIMIT_RPM)
        self._rate_limit_window = 60.0
        self._request_timestamps: list[float] = []

        self._client = None
        self._fallback_client = None
        if Config.GEMINI_API_KEY:
            self._client = genai.Client(api_key=Config.GEMINI_API_KEY)
            # Fallback provider: same key, cheaper/faster model
            if Config.GEMINI_FALLBACK_MODEL != Config.GEMINI_MODEL:
                self._fallback_model = Config.GEMINI_FALLBACK_MODEL
            else:
                self._fallback_model = None
        else:
            self._fallback_model = None

    def _check_rate_limit(self) -> None:
        now = time.monotonic()
        self._request_timestamps = [
            ts for ts in self._request_timestamps if now - ts < self._rate_limit_window
        ]
        if len(self._request_timestamps) >= self._max_requests_per_minute:
            raise RateLimitExceededError("Request rate limit reached")
        self._request_timestamps.append(now)

    async def generate(
        self,
        prompt: str,
        context: str = "",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a response with retry, rate limiting, and provider fallback."""
        temp = temperature if temperature is not None else Config.TEMPERATURE
        last_error: Exception | None = None

        rate_limited = False
        for attempt in range(1, self._max_retries + 1):
            try:
                self._check_rate_limit()
                return await self._call_provider(prompt, context, temp)
            except RateLimitExceededError as e:
                rate_limited = True
                logger.warning("Rate limit hit (attempt %d). Backing off.", attempt)
                await asyncio.sleep(self._retry_delay * attempt)
                last_error = e
            except Exception as e:
                logger.warning("Provider generation failed (attempt %d): %s", attempt, e)
                await asyncio.sleep(self._retry_delay * attempt)
                last_error = e

        # Distinct error for rate-limit exhaustion so the backend can return 429
        # instead of pretending a provider outage occurred.
        if rate_limited:
            raise RateLimitExceededError(
                f"Rate limit exceeded after {self._max_retries} attempts ({self._max_requests_per_minute} req/min)."
            )

        fallback = self.handle_fallback(prompt, context)
        raise AllProvidersFailedError(
            f"All LLM attempts failed after {self._max_retries} retries. Fallback: {fallback}. "
            f"Last error: {last_error}"
        ) from last_error

    async def _call_provider(self, prompt: str, context: str, temp: float) -> dict[str, Any]:
        """Try primary then fallback model. Gracefully degrades if all fail."""
        if self._client is None:
            self.handle_fallback(prompt, context)
            raise AllProvidersFailedError("No API key configured")

        full_prompt = self._build_prompt(prompt, context)
        gen_config = types.GenerateContentConfig(
            temperature=temp,
            top_p=Config.TOP_P,
            max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            system_instruction=Config.SYSTEM_PROMPT,
        )

        models_to_try = [Config.GEMINI_MODEL]
        if self._fallback_model and self._fallback_model not in models_to_try:
            models_to_try.append(self._fallback_model)

        last_exc: Exception | None = None
        for model in models_to_try:
            try:
                resp = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=model,
                    contents=full_prompt,
                    config=gen_config,
                )
                text = resp.text if resp.text else ""
                if not text.strip():
                    raise RuntimeError(f"Empty response from model {model}")
                return {"text": text, "provider": "gemini", "model": model}
            except Exception as e:
                logger.warning("Model '%s' failed: %s", model, e)
                last_exc = e
                continue

        self.handle_fallback(prompt, context)
        raise AllProvidersFailedError(f"All gemini models failed: {last_exc}") from last_exc

    def handle_fallback(self, prompt: str, context: str) -> str:
        """Graceful degradation when no provider is available."""
        logger.error("All providers failed. Returning degraded response.")
        return (
            "⚠️ I'm currently experiencing an overload (all AI providers are temporarily unavailable). "
            "Please try again in a moment."
        )

    def _build_prompt(self, prompt: str, context: str) -> str:
        parts = []
        if context:
            parts.append(f"Context from knowledge base:\n---\n{context}\n---\n")
        parts.append(f"User query: {prompt}")
        return "\n\n".join(parts)

    def _providers_ready(self) -> bool:
        return self._client is not None
