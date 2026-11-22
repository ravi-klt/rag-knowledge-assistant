from sklearn.metrics.pairwise import cosine_similarity

from embeddings import add_embeddings, create_embeddings
from models import KnowledgeChunk, RetrievedChunk


def retrieve_similar_records(
    query: str,
    chunks: list[KnowledgeChunk],
    limit: int = 3,
    minimum_score: float = 0.25,
) -> list[RetrievedChunk]:
    embedded = add_embeddings(chunks)
    if embedded is None:
        return []

    vectorizer, record_embeddings = embedded
    query_embedding = create_embeddings(query, vectorizer)
    scores = cosine_similarity(query_embedding, record_embeddings).ravel()
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: float(scores[index]), reverse=True
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score < minimum_score:
            break
        results.append(
            RetrievedChunk(chunk=chunks[index], similarity=score)
        )
        if len(results) >= limit:
            break
    return results
