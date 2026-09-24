from app.llm import LLM
from app.prompts import SYSTEM_PROMPT, build_user_prompt, is_refusal
from app.retrievers import Retriever
from app.schemas import AskResponse


class RagService:
    def __init__(self, retriever: Retriever, llm: LLM):
        self._retriever = retriever
        self._llm = llm

    def ask(self, question: str) -> AskResponse:
        snippets = self._retriever.retrieve(question)
        answer = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, snippets))
        return AskResponse(
            answer=answer,
            sources=[s.id for s in snippets],
            found_in_context=not is_refusal(answer),
        )
