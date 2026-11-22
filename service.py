import os
from pathlib import Path

from analytics import retrieve_similar_records
from evaluation import evaluate_retrieval
from knowledge_base import KnowledgeBase
from models import (
    DocumentRequest,
    EvaluationReport,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from parsers import load_evaluation_cases
from rag import generate_answer


class RagService:
    def __init__(self, knowledge_base: KnowledgeBase, evaluation_path: Path) -> None:
        self.knowledge_base = knowledge_base
        self.evaluation_path = evaluation_path

    def add_document(self, request: DocumentRequest) -> tuple[str, int]:
        chunks = self.knowledge_base.add_document(
            request.title, request.source, request.text
        )
        return chunks[0].document_id or "", len(chunks)

    def query(self, request: QueryRequest) -> QueryResponse:
        results = retrieve_similar_records(
            request.question,
            self.knowledge_base.list_chunks(),
            limit=request.top_k,
            minimum_score=request.min_score,
        )
        if not results:
            return QueryResponse(
                answer="Insufficient evidence: no relevant knowledge-base records were found.",
                evidence_status="NO_EVIDENCE",
                generation_mode="not_run",
                sources=[],
            )

        answer = generate_answer(request.question, results)

        return QueryResponse(
            answer=answer,
            evidence_status="SUFFICIENT_EVIDENCE",
            generation_mode=self.generation_mode,
            sources=[
                SourceResponse(
                    id=result.chunk.id,
                    title=result.chunk.title,
                    source=result.chunk.source,
                    score=round(result.similarity, 4),
                    excerpt=result.chunk.text[:500],
                )
                for result in results
            ],
        )

    def evaluate(self) -> EvaluationReport:
        cases = load_evaluation_cases(self.evaluation_path)
        return evaluate_retrieval(cases, self.knowledge_base.list_chunks())

    @property
    def generation_mode(self) -> str:
        return "openai" if os.getenv("OPENAI_API_KEY") else "extractive"
