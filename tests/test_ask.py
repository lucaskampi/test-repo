from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.knowledge_base import KNOWLEDGE_BASE
from app.llm import FakeLLM, LLMTimeoutError, LLMUnavailableError
from app.prompts import REFUSAL
from app.retrievers import FullContextRetriever, VectorRetriever
from main import app, get_llm, get_retriever


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(answer="BEON.tech offers IT staff augmentation services.")


@pytest.fixture
def client(fake_llm: FakeLLM) -> Iterator[TestClient]:
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_retriever] = lambda: FullContextRetriever()
    with TestClient(app) as c:  # runs the lifespan, so startup is covered too
        yield c
    app.dependency_overrides.clear()


def ask(client: TestClient, question: str):
    return client.post("/ask", json={"question": question})


# --- API contract ---


def test_valid_question_returns_answer_sources_and_flag(client: TestClient):
    response = ask(client, "What services does BEON offer?")

    assert response.status_code == 200
    assert response.json() == {
        "answer": "BEON.tech offers IT staff augmentation services.",
        "sources": ["kb-1", "kb-2", "kb-3"],
        "found_in_context": True,
    }


@pytest.mark.parametrize("body", [{"question": ""}, {}, {"question": "x" * 501}])
def test_invalid_question_returns_422(client: TestClient, body: dict):
    assert client.post("/ask", json=body).status_code == 422


@pytest.mark.parametrize(
    "answer", [REFUSAL, f'"{REFUSAL}"', f"  {REFUSAL.lower()}\n", REFUSAL.rstrip(".")]
)
def test_refusal_sets_found_in_context_false(client: TestClient, fake_llm: FakeLLM, answer: str):
    fake_llm.answer = answer

    assert ask(client, "What is BEON's annual revenue?").json()["found_in_context"] is False


def test_llm_unavailable_returns_503(client: TestClient, fake_llm: FakeLLM):
    fake_llm.error = LLMUnavailableError("Ollama is unreachable")

    response = ask(client, "What is BEON's mission?")

    assert response.status_code == 503
    assert response.json() == {"detail": "Ollama is unreachable"}


def test_llm_timeout_returns_504(client: TestClient, fake_llm: FakeLLM):
    fake_llm.error = LLMTimeoutError("Ollama call timed out")

    assert ask(client, "What is BEON's mission?").status_code == 504


# --- Prompt ---


def test_prompt_contains_numbered_context_and_question(client: TestClient, fake_llm: FakeLLM):
    ask(client, "What is BEON's mission?")

    system, user = fake_llm.calls[-1]
    assert REFUSAL in system
    for snippet in KNOWLEDGE_BASE:
        assert f"[{snippet.id}] {snippet.text}" in user
    assert user.endswith("Question: What is BEON's mission?")


# --- Retrievers ---


def test_full_context_retriever_returns_all_snippets():
    snippets = FullContextRetriever().retrieve("anything")

    assert [s.id for s in snippets] == ["kb-1", "kb-2", "kb-3"]


class KeywordEmbedder:
    """Fake embedder: one dimension per topic, so similarity is predictable without a model."""

    TOPICS = ["mission", "services", "culture"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if t in text.lower() else 0.1 for t in self.TOPICS] for text in texts]


def test_vector_retriever_returns_top_2_most_similar_first():
    retriever = VectorRetriever(KeywordEmbedder())

    snippets = retriever.retrieve("Tell me about the culture")

    assert len(snippets) == 2
    assert snippets[0].id == "kb-3"


def test_vector_retriever_can_be_rebuilt():
    VectorRetriever(KeywordEmbedder())

    assert VectorRetriever(KeywordEmbedder()).retrieve("mission")[0].id == "kb-1"
