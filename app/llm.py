from typing import Protocol

import ollama


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
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        return response.message.content.strip()
