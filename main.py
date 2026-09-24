import os
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm import LLM, LLMTimeoutError, LLMUnavailableError, OllamaLLM
from app.rag_service import RagService
from app.retrievers import FullContextRetriever, Retriever
from app.schemas import AskRequest, AskResponse

app = FastAPI(title="BEON.tech Q&A")


@lru_cache
def get_llm() -> LLM:
    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        timeout=float(os.getenv("OLLAMA_TIMEOUT", "60")),
    )


@lru_cache
def get_retriever() -> Retriever:
    return FullContextRetriever()


def get_rag_service(
    retriever: Retriever = Depends(get_retriever),
    llm: LLM = Depends(get_llm),
) -> RagService:
    return RagService(retriever, llm)


@app.exception_handler(LLMUnavailableError)
def llm_unavailable_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMTimeoutError)
def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    return service.ask(payload.question)
