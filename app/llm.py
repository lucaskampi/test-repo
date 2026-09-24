from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import httpx
import ollama


class LLMUnavailableError(Exception):
    """The LLM backend is unreachable or the model is not available."""


class LLMTimeoutError(Exception):
    """The LLM call exceeded the configured timeout."""


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@contextmanager
def _translate_ollama_errors(model: str) -> Iterator[None]:
    try:
        yield
    # ConnectTimeout means the server never answered, so it counts as unreachable.
    except (ConnectionError, httpx.ConnectTimeout) as e:
        raise LLMUnavailableError("Ollama is unreachable") from e
    except httpx.TimeoutException as e:
        raise LLMTimeoutError("Ollama call timed out") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise LLMUnavailableError(f"Model '{model}' is not available") from e
        raise LLMUnavailableError(f"Ollama error: {e.error}") from e


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self._model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    def generate(self, system: str, user: str) -> str:
        with _translate_ollama_errors(self._model):
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0},
            )
        return response.message.content.strip()


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self._model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with _translate_ollama_errors(self._model):
            response = self._client.embed(model=self._model, input=texts)
        return [list(v) for v in response.embeddings]


class FakeLLM:
    """Deterministic LLM for tests: returns a fixed answer or raises a given error."""

    def __init__(self, answer: str = "", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.answer
