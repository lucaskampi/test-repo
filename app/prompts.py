from app.schemas import Snippet

REFUSAL = "I don't have that information."

SYSTEM_PROMPT = (
    "You answer questions about BEON.tech using only the provided context. "
    f"If the answer is not in the context, reply exactly: {REFUSAL}"
)


def build_user_prompt(question: str, snippets: list[Snippet]) -> str:
    context = "\n".join(f"[{s.id}] {s.text}" for s in snippets)
    return f"Context:\n{context}\n\nQuestion: {question}"
