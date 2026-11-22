import threading
import uuid
from pathlib import Path

from models import KnowledgeChunk
from parsers import load_knowledge_base, save_knowledge_base, split_into_chunks


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._chunks = load_knowledge_base(path)

    def list_chunks(self) -> list[KnowledgeChunk]:
        with self._lock:
            return list(self._chunks)

    def add_document(self, title: str, source: str, text: str) -> list[KnowledgeChunk]:
        text_chunks = split_into_chunks(text)
        if not text_chunks:
            raise ValueError("Document text must not be blank.")

        document_id = str(uuid.uuid4())
        chunks = [
            KnowledgeChunk(
                id=f"{document_id}-{index + 1}",
                document_id=document_id,
                title=title,
                source=source,
                text=chunk,
            )
            for index, chunk in enumerate(text_chunks)
        ]
        with self._lock:
            updated_chunks = [*self._chunks, *chunks]
            save_knowledge_base(self.path, updated_chunks)
            self._chunks = updated_chunks
        return chunks
