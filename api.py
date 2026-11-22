import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from knowledge_base import KnowledgeBase
from models import DocumentRequest, QueryRequest
from service import RagService

logger = logging.getLogger(__name__)


def create_app(
    data_path: Path | None = None, evaluation_path: Path | None = None
) -> FastAPI:
    project_dir = Path(__file__).resolve().parent
    knowledge_base = KnowledgeBase(
        data_path or project_dir / "data" / "knowledge_base.json"
    )
    service = RagService(
        knowledge_base,
        evaluation_path or project_dir / "data" / "evaluation.json",
    )
    application = FastAPI(
        title="Incident RAG POC",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.knowledge_base = knowledge_base
    application.state.rag_service = service

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "UP",
            "chunks": len(knowledge_base.list_chunks()),
            "generation": service.generation_mode,
        }

    @application.get("/documents")
    def list_documents() -> dict[str, Any]:
        return {"documents": knowledge_base.list_chunks()}

    @application.post("/documents", status_code=201)
    def add_document(request: DocumentRequest) -> dict[str, Any]:
        document_id, chunks_added = service.add_document(request)
        return {"document_id": document_id, "chunks_added": chunks_added}

    @application.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        try:
            return service.query(request).model_dump()
        except Exception as error:
            logger.exception("RAG query failed")
            raise HTTPException(
                status_code=502,
                detail="RAG query failed; check server logs.",
            ) from error

    @application.get("/evaluation")
    def run_evaluation() -> dict[str, Any]:
        try:
            return service.evaluate().model_dump()
        except Exception as error:
            logger.exception("RAG evaluation failed")
            raise HTTPException(
                status_code=500,
                detail="RAG evaluation failed; check server logs.",
            ) from error

    return application


app = create_app()
