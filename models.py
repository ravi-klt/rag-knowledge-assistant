from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(default="user-upload", min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100_000)

    @field_validator("title", "source", "text")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must not be blank.")
        return normalized


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question must not be blank.")
        return normalized


class KnowledgeChunk(BaseModel):
    id: str
    document_id: str | None = None
    title: str
    source: str
    text: str


class RetrievedChunk(BaseModel):
    chunk: KnowledgeChunk
    similarity: float


class SourceResponse(BaseModel):
    id: str
    title: str
    source: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    evidence_status: Literal["NO_EVIDENCE", "SUFFICIENT_EVIDENCE"]
    generation_mode: Literal["openai", "extractive", "not_run"]
    sources: list[SourceResponse]


class EvaluationCase(BaseModel):
    query: str
    expected_incident_id: str | None


class EvaluationResult(BaseModel):
    query: str
    expected_incident_id: str | None
    actual_id: str | None
    similarity: float
    correct: bool


class EvaluationReport(BaseModel):
    accuracy: float
    total_cases: int
    correct_cases: int
    results: list[EvaluationResult]
