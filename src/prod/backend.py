from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config
from src.prod.cache import TTLCache
from src.prod.llm_service import AllProvidersFailedError, LLMService, RateLimitExceededError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic request/response models ---------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    structured: bool = Field(False, description="Request JSON structured output")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    use_cache: bool = Field(True, description="Use prompt/response cache")


class IngestionRequest(BaseModel):
    source: str = Field(..., min_length=1, description="File or directory path to ingest")


class ChatResponse(BaseModel):
    message: str
    response: str
    provider: str | None = None
    context_used: bool = False
    cached: bool = False
    error: str | None = None


class ResearchRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User query")
    reply: str | None = Field(None, description="Answer to a pending clarification (multi-turn)")


class ResearchResponse(BaseModel):
    message: str
    response: str
    confidence: str
    status: str
    steps: int
    degraded: bool
    sources: list[str]
    tool_calls: list[str]
    token_total: int
    pending_question: str | None = None


class StatusResponse(BaseModel):
    status: str
    vector_store_chunks: int
    cache_size: int
    provider: str
    model: str


# --- Application state -------------------------------------------------------

class AppState:
    def __init__(self):
        self.llm_service = LLMService()
        self.cache = TTLCache(
            ttl_seconds=Config.CACHE_TTL,
            max_size=Config.CACHE_MAX_SIZE,
        )
        self.vector_store = None
        self._rag_disabled = True


app_state = AppState()


def _retrieve_context(query: str) -> str:
    if app_state.vector_store is None:
        return ""
    try:
        results = app_state.vector_store.query(query, top_k=Config.TOP_K)
        if not results:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {r['metadata'].get('source', 'unknown')}]\n{r['content']}"
            for r in results
        )
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        return ""


def _build_structured_prompt(prompt: str, schema: dict) -> str:
    import json as _json
    schema_str = _json.dumps(schema, indent=2)
    return (
        f"{prompt}\n\n"
        f"Please respond with valid JSON matching this schema:\n{schema_str}"
    )


STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence"],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Assistant backend...")
    # Lazily initialize vector store if data directory exists
    try:
        from src.rag.vector_store import VectorStore
        vs = VectorStore()
        if vs.count > 0:
            app_state.vector_store = vs
            app_state._rag_disabled = False
            logger.info("RAG enabled with %d chunks", vs.count)
        else:
            logger.info("Vector store empty; RAG disabled until ingestion")
    except Exception as e:
        logger.warning("Could not initialize vector store: %s", e)
    yield
    logger.info("Shutting down backend.")


app = FastAPI(title="AI Assistant Backend", version="2.0.0", lifespan=lifespan)

# CORS for the Streamlit UI / web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse, tags=["system"])
async def status():
    return StatusResponse(
        status="ok" if app_state.llm_service._providers_ready() else "degraded",
        vector_store_chunks=app_state.vector_store.count if app_state.vector_store else 0,
        cache_size=app_state.cache.size,
        provider="gemini",
        model=Config.GEMINI_MODEL,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """Main chat endpoint. Handles concurrency, retry, fallback, caching."""
    context = _retrieve_context(request.message)

    temperature = request.temperature if request.temperature is not None else Config.TEMPERATURE

    # Prompt/response caching
    cached_response = None
    if request.use_cache:
        cached_response = app_state.cache.get(request.message, context, temperature)

    if cached_response is not None:
        return ChatResponse(
            message=request.message,
            response=cached_response["text"],
            provider=cached_response.get("provider"),
            context_used=bool(context),
            cached=True,
        )

    # Build the final prompt (structured or plain)
    if request.structured:
        final_prompt = _build_structured_prompt(request.message, STRUCTURED_SCHEMA)
    else:
        final_prompt = request.message

    try:
        result = await app_state.llm_service.generate(
            final_prompt,
            context=context,
            temperature=temperature,
        )
    except RateLimitExceededError as e:
        logger.error("Rate limited: %s", e)
        raise HTTPException(
            status_code=429, detail="Too many requests. Please wait and retry."
        ) from e
    except AllProvidersFailedError as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=503, detail="All AI providers unavailable") from e

    if request.use_cache:
        app_state.cache.set(request.message, context, temperature, result)

    return ChatResponse(
        message=request.message,
        response=result["text"],
        provider=result.get("provider"),
        context_used=bool(context),
        cached=False,
    )


_RESEARCH_SESSIONS: dict[str, Any] = {}


@app.post("/research", response_model=ResearchResponse, tags=["agentic"])
async def research(request: ResearchRequest):
    """Agentic research loop. Multi-turn: if the agent returns
    `status == "clarification_pending"`, re-call with `reply` to continue.

    Robustness rules for out-of-band/duplicate calls:
    - `reply` with no active session  -> 404
    - `reply` on a session that already finalized -> 409 (no pending question)
    - `reply` while the agent is not waiting -> 409
    - no `reply`, but a previous call left the agent waiting on a
      clarification -> the pending question is returned again (session kept)."""
    from src.agents.research_agent import ResearchSession
    from src.rag.vector_store import VectorStore

    session = _RESEARCH_SESSIONS.get(request.message)

    if request.reply:
        if session is None:
            raise HTTPException(status_code=404, detail="No active session for this query")
        if session.is_finalized:
            raise HTTPException(
                status_code=409,
                detail="This research is already complete; start a new query",
            )
        if session.pending_question is None:
            raise HTTPException(
                status_code=409,
                detail="The agent is not currently waiting for a clarification",
            )
        result = session.continue_with_reply(request.reply)
    elif session is not None and session.pending_question is not None:
        result = session.to_result()
    else:
        session = ResearchSession(
            request.message,
            vector_store=app_state.vector_store or VectorStore(),
        )
        _RESEARCH_SESSIONS[request.message] = session
        result = session.run()

    return ResearchResponse(
        message=request.message,
        response=result["answer"] or result.get("pending_question", ""),
        confidence=result.get("confidence", "") if result["status"] == "finalized" else "",
        status=result["status"],
        steps=result["steps"],
        degraded=result["degraded"],
        sources=[str(s) for s in result.get("sources", [])],
        tool_calls=[t["action"] for t in result["tool_calls"]],
        token_total=result["tokens"]["total_tokens"],
        pending_question=result.get("pending_question"),
    )


@app.post("/ingest", tags=["rag"])
async def ingest(request: IngestionRequest):
    """Ingest documents into the RAG knowledge base."""
    from src.rag.document_loader import chunk_documents, load_documents
    from src.rag.vector_store import VectorStore

    try:
        docs = load_documents(request.source)
        chunks = chunk_documents(docs)
        vs = VectorStore()
        count = vs.add_documents(chunks)
        app_state.vector_store = vs
        app_state._rag_disabled = False
        return {"ingested": count, "source": request.source}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
