from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from models import KnowledgeChunk


def add_embeddings(
    chunks: list[KnowledgeChunk],
) -> tuple[TfidfVectorizer, csr_matrix] | None:
    if not chunks:
        return None

    texts = [f"{chunk.title} {chunk.text}" for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        vectors = vectorizer.fit_transform(texts)
    except ValueError:
        return None
    return vectorizer, vectors


def create_embeddings(query: str, vectorizer: TfidfVectorizer) -> csr_matrix:
    return vectorizer.transform([query])
