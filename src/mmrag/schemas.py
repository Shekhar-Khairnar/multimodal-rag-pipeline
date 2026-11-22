"""Structured response returned by the RAG pipeline (used by the UI, API and evaluation)."""

from pydantic import BaseModel


class Source(BaseModel):
    number: int  # citation number used in the answer, e.g. [1]
    source: str  # file name
    page: int
    modality: str  # "text" | "table" | "visual"
    score: float  # retrieval similarity
    chunk_id: str
    content: str
    image_path: str | None = None  # set for visuals


class RAGResponse(BaseModel):
    question: str
    answer: str
    route: str  # "text" (text LLM) or "visual" (VLM with the original images)
    model: str
    sources: list[Source]
    timings: dict[str, float]  # seconds: retrieval, generation, total
    usage: dict  # token counts reported by the model
