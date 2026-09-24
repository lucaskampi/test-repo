from functools import lru_cache

from fastapi import Depends, FastAPI

from app.llm import LLM, OllamaLLM
from app.rag_service import RagService
from app.retrievers import FullContextRetriever, Retriever
from app.schemas import AskRequest, AskResponse

app = FastAPI(title="BEON.tech Q&A")


@lru_cache
def get_llm() -> LLM:
    return OllamaLLM()


@lru_cache
def get_retriever() -> Retriever:
    return FullContextRetriever()


def get_rag_service(
    retriever: Retriever = Depends(get_retriever),
    llm: LLM = Depends(get_llm),
) -> RagService:
    return RagService(retriever, llm)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, service: RagService = Depends(get_rag_service)) -> AskResponse:
    return service.ask(payload.question)
