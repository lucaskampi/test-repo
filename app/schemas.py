from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    found_in_context: bool


class Snippet(BaseModel):
    id: str
    text: str
