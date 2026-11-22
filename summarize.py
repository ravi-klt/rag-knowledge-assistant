import re

from models import RetrievedChunk


def build_rag_context(results: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{result.chunk.id}] {result.chunk.title}: {result.chunk.text}"
        for result in results
    )


def build_extractive_answer(results: list[RetrievedChunk]) -> str:
    excerpts = []
    for result in results:
        text = re.sub(r"\s+", " ", result.chunk.text).strip()
        excerpts.append(f"[{result.chunk.id}] {result.chunk.title}: {text}")
    return "Relevant evidence:\n" + "\n".join(excerpts)
