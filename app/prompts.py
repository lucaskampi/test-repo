from app.schemas import Snippet

REFUSAL = "I don't have that information."

SYSTEM_PROMPT = f"""You are a Q&A assistant for BEON.tech.

Rules:
- Answer using ONLY the information in the context below. Never use outside knowledge.
- If the context does not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}
- Answer in at most 2 sentences of plain prose. No lists, no markdown.
- Only state facts written in the context. Do not add interpretations, opinions or filler.
- Start with the answer itself. No preamble such as "According to the context".
- Do not mention the context or the snippet IDs."""


def build_user_prompt(question: str, snippets: list[Snippet]) -> str:
    context = "\n".join(f"[{s.id}] {s.text}" for s in snippets)
    return f"Context:\n{context}\n\nQuestion: {question}"


def is_refusal(answer: str) -> bool:
    """Tolerates quotes, whitespace and case, which small models sometimes add."""
    normalized = answer.strip().strip("\"'").strip().lower()
    return normalized.startswith(REFUSAL.lower().rstrip("."))
