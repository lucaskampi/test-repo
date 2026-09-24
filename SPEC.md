# SPEC: BEON.tech Q&A service (RAG)

## Goal
An HTTP service that answers questions about BEON.tech using only a small knowledge base,
with a local open-source LLM (no API key). The answer must never go beyond the provided context.

## Stack
- Python 3.12, FastAPI, uv
- LLM: Ollama `llama3.2:3b` at `http://localhost:11434` (local, no API key)
- Embeddings: Ollama `nomic-embed-text` (768 dims)
- Vector DB: ChromaDB, embedded and in memory, cosine distance

## API contract
`POST /ask`

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

class AskResponse(BaseModel):
    answer: str
    sources: list[str]        # snippet IDs used as context, e.g. ["kb-2"]
    found_in_context: bool    # False when the model says it doesn't know
```

| Status | When |
|---|---|
| 200 | Answer returned (including "I don't have that information.") |
| 422 | Invalid request (missing or empty question) |
| 503 | Ollama unreachable or model not available |
| 504 | LLM call exceeded the timeout |

Example:
```json
// POST /ask {"question": "What services does BEON offer?"}
{
  "answer": "BEON.tech offers IT staff augmentation services, from backend and frontend to AI, machine learning, DevOps and QA.",
  "sources": ["kb-2"],
  "found_in_context": true
}
```

## Knowledge base
Stored as data with stable IDs, so answers can cite their sources:
- `kb-1`: mission
- `kb-2`: services
- `kb-3`: culture

## Flow
1. FastAPI validates the request against `AskRequest`.
2. `RagService.ask(question)` calls the retriever to get the relevant snippets.
3. The prompt builder puts the snippets into a numbered context (`[kb-2] ...`) and adds the question.
4. The LLM client calls Ollama with `temperature=0` and a timeout.
5. The service builds `AskResponse`: `sources` = IDs of the snippets sent, `found_in_context` = answer is not the refusal sentence.

## Key decisions
1. **Retriever interface, two implementations.** `retrieve(question) -> list[Snippet]`:
   - `FullContextRetriever` returns all snippets (core task).
   - `VectorRetriever` embeds the question and queries Chroma, top-k = 2 (bonus).
   Selected by config (`RETRIEVER=full|vector`). The service, prompt and API don't change.
2. **Prompt against hallucination:**
   - System instruction: answer only from the context.
   - An explicit way out: if the answer isn't in the context, reply exactly `I don't have that information.`
   - At most 2 sentences, no preamble.
   - `temperature=0`, so the output stays deterministic and faithful.
3. **Same embedding model for indexing and querying.** Mixing models makes the vectors incomparable.
4. **Index at startup.** Three snippets are embedded into an in-memory Chroma collection
   (`knowledge_base`, cosine) when the app starts. No stale data between runs.
5. **Warm up the model at startup.** The first call to `llama3.2:3b` took ~50s (model loading);
   later calls took ~1.5s. One warm-up call on startup keeps the first real request fast.
6. **LLM client injected via `Depends`.** Tests use a fake LLM: no network, deterministic,
   and switching providers touches one file.
7. **Timeout on Ollama calls**, mapped to 504. A clear error is better than a hanging request.

## Structure
```
main.py            # FastAPI app, POST /ask, startup (index + warm-up), dependency wiring
app/
  schemas.py       # AskRequest, AskResponse, Snippet
  knowledge_base.py# the three snippets with IDs
  retrievers.py    # Retriever protocol, FullContextRetriever, VectorRetriever
  llm.py           # OllamaLLM (+ FakeLLM for tests)
  prompts.py       # system prompt + context/question builder
  rag_service.py   # orchestrates retrieve → prompt → LLM → response
tests/
  test_ask.py
```

## Tests (fake LLM, no network)
- Valid question returns 200 with `answer`, `sources`, `found_in_context`.
- Empty question returns 422.
- Refusal answer gives `found_in_context = false`.
- `FullContextRetriever` returns all 3 snippets.

Manual checks against the real model:
| Question | Expected |
|---|---|
| What services does BEON offer? | Answer from `kb-2` |
| What is BEON's mission? | Answer from `kb-1` |
| What is BEON's annual revenue? | `I don't have that information.` |

## Implementation steps
1. Schemas, knowledge base and `POST /ask` using `FullContextRetriever` + Ollama. Check with curl.
2. Prompt rules + `found_in_context`, verified with the three manual questions.
3. Error handling (503/504) and timeout.
4. Tests with a fake LLM.
5. Bonus: `VectorRetriever` with Chroma + embeddings, switched by config. Re-run the manual checks.

## Out of scope (and how I'd add it)
- **Distance threshold** in retrieval: drop snippets above a cosine distance, since "most similar" can still be irrelevant. Needs tuning on real data.
- **Persistence and larger corpora**: Chroma `PersistentClient` or pgvector (vectors next to relational data, SQL filters); chunking for long documents.
- **Evals**: a labeled set of questions (answerable and unanswerable), scoring accuracy and refusal rate on every prompt or model change.
- **Observability**: log latency, tokens and retrieved IDs per request.
- **Streaming responses** and **Docker packaging** for deployment.

## Open questions
- Should the refusal be a fixed sentence, or a flag with an empty answer?
- Any latency target? A local 3B model on CPU/GPU is slower than a hosted API.
- Is English-only fine for questions?
