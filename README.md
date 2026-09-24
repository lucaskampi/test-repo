# BEON.tech Q&A service

An HTTP service that answers questions about BEON.tech using only a small knowledge base (RAG),
with a local open-source LLM through Ollama — no API key. When the answer isn't in the knowledge
base, it replies `I don't have that information.` instead of guessing.

See [SPEC.md](SPEC.md) for the full design and decisions.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12
- [Ollama](https://ollama.com/download) running locally, with the two models pulled:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn main:app --reload
```

On startup the app indexes the knowledge base and makes one warm-up call, so the model is loaded
before the first request. Interactive docs: http://localhost:8000/docs

## Try it

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does BEON offer?"}'
```

Response:

```json
{
  "answer": "BEON.tech offers IT staff augmentation services, providing engineers for backend, frontend, AI, machine learning, DevOps and QA roles.",
  "sources": ["kb-1", "kb-2", "kb-3"],
  "found_in_context": true
}
```

- `sources`: IDs of the knowledge base snippets sent to the model as context.
- `found_in_context`: `false` when the model replies `I don't have that information.`

| Status | When |
|---|---|
| 200 | Answer returned (including the refusal) |
| 422 | Invalid request (missing, empty or over 500 characters) |
| 503 | Ollama unreachable or model not available |
| 504 | LLM call exceeded the timeout |

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `RETRIEVER` | `full` | `full` sends every snippet; `vector` embeds the question and returns the top 2 from Chroma |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (used when `RETRIEVER=vector`) |
| `OLLAMA_TIMEOUT` | `60` | Seconds before an Ollama call returns 504 |

```bash
RETRIEVER=vector uv run uvicorn main:app --reload
```

In `vector` mode the app won't start if Ollama is down, because it can't build the index.
In `full` mode it starts anyway and requests return 503 until Ollama is back.

## Tests

The tests use a fake LLM and a fake embedder, so they run in about a second without Ollama:

```bash
uv run pytest
```

They cover the API contract (200/422/503/504), refusal detection, the prompt, and both retrievers.

## Project structure

```
main.py              # FastAPI app, POST /ask, startup (index + warm-up), dependency wiring
app/
  schemas.py         # AskRequest, AskResponse, Snippet
  knowledge_base.py  # the three snippets with IDs (kb-1 mission, kb-2 services, kb-3 culture)
  retrievers.py      # Retriever protocol, FullContextRetriever, VectorRetriever
  llm.py             # OllamaLLM, OllamaEmbedder, FakeLLM, error mapping
  prompts.py         # system prompt, context/question builder, refusal check
  rag_service.py     # retrieve → prompt → LLM → response
tests/
  test_ask.py
```

## Not done

Evals were skipped for the sake of time. The plan was a labelled set of answerable and
unanswerable questions, run against the real model on every prompt or model change, scoring
answer accuracy, refusal rate and retrieval hit rate. The fake-LLM tests check the code, not
the model: a 3B model can still add small unsupported details (for example, extra claims about
culture in `vector` mode), and only evals would catch that over time.
