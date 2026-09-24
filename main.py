import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm import LLM, LLMTimeoutError, LLMUnavailableError, OllamaEmbedder, OllamaLLM
from app.rag_service import RagService
from app.retrievers import FullContextRetriever, Retriever, VectorRetriever
from app.schemas import AskRequest, AskResponse

logger = logging.getLogger("uvicorn.error")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


@lru_cache
def get_llm() -> LLM:
    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT,
    )


@lru_cache
def get_retriever() -> Retriever:
    kind = os.getenv("RETRIEVER", "full")
    if kind == "full":
        return FullContextRetriever()
    if kind == "vector":
        embedder = OllamaEmbedder(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT,
        )
        return VectorRetriever(embedder)
    raise ValueError(f"RETRIEVER must be 'full' or 'vector', got '{kind}'")


def get_rag_service(
    retriever: Retriever = Depends(get_retriever),
    llm: LLM = Depends(get_llm),
) -> RagService:
    return RagService(retriever, llm)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Index at startup: the vector retriever needs Ollama, so a failure here stops the app.
    retriever = get_retriever()
    logger.info("Retriever ready: %s", type(retriever).__name__)
    # Warm up: the first call loads the model into memory, which can take ~50s.
    try:
        get_llm().generate("Reply with OK.", "OK")
        logger.info("LLM warmed up")
    except (LLMUnavailableError, LLMTimeoutError) as e:
        logger.warning("LLM warm-up failed, requests will return 503/504: %s", e)
    yield


app = FastAPI(title="BEON.tech Q&A", lifespan=lifespan)


@app.exception_handler(LLMUnavailableError)
def llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMTimeoutError)
def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    return service.ask(payload.question)
