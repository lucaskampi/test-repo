from typing import Protocol

import httpx
import ollama


class LLMUnavailableError(Exception):
    """The LLM backend is unreachable or the model is not available."""


class LLMTimeoutError(Exception):
    """The LLM call exceeded the configured timeout."""


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


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
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0},
            )
        # ConnectTimeout means the server never answered, so it counts as unreachable.
        except (ConnectionError, httpx.ConnectTimeout) as e:
            raise LLMUnavailableError("Ollama is unreachable") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Ollama call timed out") from e
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMUnavailableError(f"Model '{self._model}' is not available") from e
            raise LLMUnavailableError(f"Ollama error: {e.error}") from e
        return response.message.content.strip()
