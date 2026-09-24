from app.schemas import Snippet

KNOWLEDGE_BASE: list[Snippet] = [
    Snippet(
        id="kb-1",
        text=(
            "BEON.tech's mission is to connect US companies with top software engineers "
            "from Latin America, helping them build high-performing remote teams."
        ),
    ),
    Snippet(
        id="kb-2",
        text=(
            "BEON.tech offers IT staff augmentation services, providing engineers for "
            "backend, frontend, AI, machine learning, DevOps and QA roles."
        ),
    ),
    Snippet(
        id="kb-3",
        text=(
            "BEON.tech's culture is remote-first and built on continuous learning, "
            "transparency and long-term relationships with its engineers and clients."
        ),
    ),
]
