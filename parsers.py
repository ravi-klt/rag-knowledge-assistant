import json
import os
import tempfile
from pathlib import Path

from models import EvaluationCase, KnowledgeChunk


def load_knowledge_base(path: Path) -> list[KnowledgeChunk]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Knowledge base must contain a JSON list: {path}")
    return [KnowledgeChunk.model_validate(record) for record in records]


def save_knowledge_base(path: Path, chunks: list[KnowledgeChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as file:
            temporary_path = file.name
            json.dump(
                [chunk.model_dump() for chunk in chunks],
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Evaluation file must contain a JSON list: {path}")
    return [EvaluationCase.model_validate(record) for record in records]


def split_into_chunks(
    text: str, chunk_size: int = 900, overlap: int = 120
) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        if current and current_length + len(word) + 1 > chunk_size:
            chunks.append(" ".join(current))
            overlap_words: list[str] = []
            overlap_length = 0
            for previous_word in reversed(current):
                if overlap_length + len(previous_word) + 1 > overlap:
                    break
                overlap_words.insert(0, previous_word)
                overlap_length += len(previous_word) + 1
            current = overlap_words
            current_length = overlap_length
        current.append(word)
        current_length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks
