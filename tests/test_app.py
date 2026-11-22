from pathlib import Path

from fastapi.testclient import TestClient

from api import create_app
from evaluation import evaluate_retrieval
from parsers import load_evaluation_cases, load_knowledge_base


def client_for(tmp_path):
    return TestClient(create_app(tmp_path / "knowledge_base.json"))


def test_docs_are_disabled_but_health_is_available(tmp_path):
    client = client_for(tmp_path)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/health").status_code == 200


def test_add_and_query_document_with_citation(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = client_for(tmp_path)
    added = client.post(
        "/documents",
        json={
            "title": "Payment incident",
            "source": "test",
            "text": (
                "Payment API returned HTTP 500 because the database connection "
                "pool was exhausted. Resolution: increase the pool size."
            ),
        },
    )
    assert added.status_code == 201

    response = client.post(
        "/query", json={"question": "Why did payment requests return HTTP 500?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["evidence_status"] == "SUFFICIENT_EVIDENCE"
    assert body["generation_mode"] == "extractive"
    assert "connection pool was exhausted" in body["answer"]
    assert body["sources"][0]["title"] == "Payment incident"
    assert body["sources"][0]["score"] > 0


def test_query_reports_openai_generation_mode_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "service.generate_answer", lambda question, results: "Generated answer [INC-1]"
    )
    client = client_for(tmp_path)
    client.post(
        "/documents",
        json={"title": "Payment", "source": "test", "text": "Payment pool exhausted."},
    )

    response = client.post("/query", json={"question": "Payment pool exhausted?"})

    assert response.status_code == 200
    assert response.json()["generation_mode"] == "openai"


def test_query_without_relevant_evidence_abstains(tmp_path):
    client = client_for(tmp_path)
    response = client.post(
        "/query", json={"question": "What is the recipe for chocolate cake?"}
    )
    assert response.status_code == 200
    assert response.json()["evidence_status"] == "NO_EVIDENCE"
    assert response.json()["generation_mode"] == "not_run"
    assert response.json()["sources"] == []


def test_ingested_documents_persist_between_app_instances(tmp_path):
    path = tmp_path / "knowledge_base.json"
    client = TestClient(create_app(path))
    response = client.post(
        "/documents",
        json={"title": "Runbook", "source": "ops", "text": "Restart the worker."},
    )
    assert response.status_code == 201

    reloaded_client = TestClient(create_app(path))
    documents = reloaded_client.get("/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["text"] == "Restart the worker."


def test_document_and_question_must_not_be_blank(tmp_path):
    client = client_for(tmp_path)
    document = client.post(
        "/documents",
        json={"title": "   ", "source": "test", "text": "Useful text"},
    )
    question = client.post("/query", json={"question": "   "})

    assert document.status_code == 422
    assert question.status_code == 422


def test_bundled_retrieval_evaluation():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    chunks = load_knowledge_base(data_dir / "knowledge_base.json")
    cases = load_evaluation_cases(data_dir / "evaluation.json")

    report = evaluate_retrieval(cases, chunks)

    assert report.total_cases == 5
    assert report.accuracy == 1.0


def test_evaluation_api_returns_report():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    client = TestClient(
        create_app(
            data_path=data_dir / "knowledge_base.json",
            evaluation_path=data_dir / "evaluation.json",
        )
    )

    response = client.get("/evaluation")

    assert response.status_code == 200
    assert response.json()["accuracy"] == 1.0
