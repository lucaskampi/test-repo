from typing import Protocol

from app.knowledge_base import KNOWLEDGE_BASE
from app.schemas import Snippet


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[Snippet]: ...


class FullContextRetriever:
    """Returns every snippet; the knowledge base is small enough to fit in the prompt."""

    def __init__(self, snippets: list[Snippet] = KNOWLEDGE_BASE):
        self._snippets = snippets

    def retrieve(self, question: str) -> list[Snippet]:
        return list(self._snippets)
