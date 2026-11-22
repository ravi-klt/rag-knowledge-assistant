# Incident RAG POC

A self-contained retrieval-augmented question-answering API for historical
incident notes and operational runbooks.
Requires Python 3.10 or newer.

## What it does

- Ingests text documents and persists them as searchable chunks.
- Retrieves relevant evidence using local TF-IDF and cosine similarity.
- Returns source IDs, scores, and excerpts with every supported answer.
- Includes `generation_mode` in each `/query` response: `openai`, `extractive`,
  or `not_run` when no evidence is retrieved.
- Abstains when no evidence meets the relevance threshold.
- Works without credentials using a grounded extractive response. Set
  `OPENAI_API_KEY` to enable optional OpenAI answer synthesis.

## Run locally

From this directory, create and activate a virtual environment, then install
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Set the OpenAI API key and run the API

Set the key in the same PowerShell terminal that will run the API. The project
does not automatically load `.env` files; do not commit or share your key.

```powershell
$secureKey = Read-Host -Prompt "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
$env:OPENAI_MODEL = "gpt-4.1-mini"
python main.py serve
```

The server listens on `http://127.0.0.1:8000`. Swagger, ReDoc, the OpenAPI
schema, and the root route are disabled. The health endpoint remains available
at `GET /health`; the RAG endpoints (`/documents`, `/query`, and `/evaluation`)
remain available for API clients such as Postman.

## Project modules

- `main.py`: command-line entry point for serving and retrieval evaluation.
- `api.py`: FastAPI endpoints for health, documents, queries, and evaluation.
- `models.py`: request, response, document, and evaluation schemas.
- `parsers.py` and `knowledge_base.py`: JSON loading, chunking, and persistence.
- `embeddings.py` and `analytics.py`: TF-IDF vector creation and retrieval.
- `rag.py` and `summarize.py`: optional OpenAI answer generation and offline
  evidence-based answer formatting.
- `evaluation.py`: top-1 retrieval accuracy, including an expected no-match
  case, against `data/evaluation.json`. The sample relevance cutoff is 0.25;
  tune it for your own documents and evaluation data.

## API examples

Add a document:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/documents `
  -ContentType 'application/json' `
  -Body '{"title":"Payment runbook","source":"ops","text":"If the payment database connection pool is exhausted, inspect pool usage and increase its configured limit."}'
```

Ask a question:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
  -ContentType 'application/json' `
  -Body '{"question":"What should I check when the payment connection pool is exhausted?","top_k":3}'
```

The sample knowledge base is in `data/knowledge_base.json`. Newly ingested
chunks are saved to the same file.

## OpenAI generation behavior

When `OPENAI_API_KEY` is set in the server process, `/query` uses OpenAI to
generate an answer from retrieved evidence. Without it, the API returns a
grounded extractive answer. The response's `generation_mode` field identifies
which path was used. If a configured OpenAI request fails, `/query` returns
HTTP 502 rather than silently switching to extractive output.

## Retrieval evaluation

Run the bundled evaluation from this directory:

```powershell
python main.py evaluate
```

The same report is available from `GET /evaluation`.

## Tests

```powershell
python -m pytest -q
```
