from typing import Protocol

import chromadb

from app.knowledge_base import KNOWLEDGE_BASE
from app.llm import Embedder
from app.schemas import Snippet


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[Snippet]: ...


class FullContextRetriever:
    """Returns every snippet; the knowledge base is small enough to fit in the prompt."""

    def __init__(self, snippets: list[Snippet] = KNOWLEDGE_BASE):
        self._snippets = snippets

    def retrieve(self, question: str) -> list[Snippet]:
        return list(self._snippets)


# nomic-embed-text is trained with task prefixes; the same model embeds both sides.
_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class VectorRetriever:
    """Embeds the snippets into an in-memory Chroma collection and returns the top-k matches."""

    def __init__(
        self,
        embedder: Embedder,
        snippets: list[Snippet] = KNOWLEDGE_BASE,
        top_k: int = 2,
    ):
        self._embedder = embedder
        self._top_k = top_k
        self._by_id = {s.id: s for s in snippets}
        self._collection = chromadb.EphemeralClient().create_collection(
            name="knowledge_base",
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self._collection.add(
            ids=[s.id for s in snippets],
            documents=[s.text for s in snippets],
            embeddings=embedder.embed([_DOCUMENT_PREFIX + s.text for s in snippets]),
        )

    def retrieve(self, question: str) -> list[Snippet]:
        [query_embedding] = self._embedder.embed([_QUERY_PREFIX + question])
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=self._top_k
        )
        return [self._by_id[snippet_id] for snippet_id in result["ids"][0]]
