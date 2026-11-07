import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

    # Production settings
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))
    RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
    CACHE_TTL: float = float(os.getenv("CACHE_TTL", "300.0"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    # Agentic settings
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", "")
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "5"))
    EVIDENCE_NOTE_MAX_CHARS: int = int(os.getenv("EVIDENCE_NOTE_MAX_CHARS", "800"))
    EVIDENCE_MAX_NOTES: int = int(os.getenv("EVIDENCE_MAX_NOTES", "6"))
    RETRIEVAL_HIT_CAP: int = int(os.getenv("RETRIEVAL_HIT_CAP", "4"))
    # Relevance buckets for retrieved chunks (cosine distance, lower = more similar).
    RETRIEVAL_RELEVANT_MAX: float = float(os.getenv("RETRIEVAL_RELEVANT_MAX", "0.42"))
    RETRIEVAL_MARGINAL_MAX: float = float(os.getenv("RETRIEVAL_MARGINAL_MAX", "0.55"))
    # Failure injection for the evaluation harness:
    #   none | tool_unavailable | malformed_retrieval | max_steps
    FAILURE_INJECTION: str = os.getenv("FAILURE_INJECTION", "none")
    # Minimum seconds between agent LLM calls (free-tier Gemini rate limit pacing).
    AGENT_PACE_SECONDS: float = float(os.getenv("AGENT_PACE_SECONDS", "0"))
    # Approximate $ per 1K tokens for cost accounting (configurable; estimates only).
    USD_PER_1K_INPUT: float = float(os.getenv("USD_PER_1K_INPUT", "0.00125"))
    USD_PER_1K_OUTPUT: float = float(os.getenv("USD_PER_1K_OUTPUT", "0.00375"))

    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        """You are a knowledgeable AI assistant with access to a document knowledge base.
You provide accurate, helpful, and well-structured responses.

When answering questions:
- Use the retrieved context when available to provide accurate answers.
- If the context doesn't contain relevant information, say so honestly.
- Structure your responses clearly.
- When asked to perform calculations or lookups, use the available tools.

Always respond in valid JSON when the user requests structured output.""",
    )

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required. "
                "Set it in .env or export it."
            )
