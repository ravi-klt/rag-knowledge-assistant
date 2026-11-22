from analytics import retrieve_similar_records
from models import EvaluationCase, EvaluationReport, EvaluationResult, KnowledgeChunk


def evaluate_retrieval(
    cases: list[EvaluationCase],
    chunks: list[KnowledgeChunk],
    minimum_score: float = 0.25,
) -> EvaluationReport:
    results = []
    for case in cases:
        retrieved = retrieve_similar_records(
            case.query, chunks, limit=1, minimum_score=minimum_score
        )
        actual_id = retrieved[0].chunk.id if retrieved else None
        similarity = retrieved[0].similarity if retrieved else 0.0
        results.append(
            EvaluationResult(
                query=case.query,
                expected_incident_id=case.expected_incident_id,
                actual_id=actual_id,
                similarity=similarity,
                correct=actual_id == case.expected_incident_id,
            )
        )

    correct_cases = sum(result.correct for result in results)
    return EvaluationReport(
        accuracy=correct_cases / len(results) if results else 0.0,
        total_cases=len(results),
        correct_cases=correct_cases,
        results=results,
    )
