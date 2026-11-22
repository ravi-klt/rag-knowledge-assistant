import argparse
import json
from pathlib import Path

import uvicorn

from evaluation import evaluate_retrieval
from knowledge_base import KnowledgeBase
from parsers import load_evaluation_cases

PROJECT_DIR = Path(__file__).resolve().parent


def run_evaluation() -> None:
    knowledge_base = KnowledgeBase(PROJECT_DIR / "data" / "knowledge_base.json")
    cases = load_evaluation_cases(PROJECT_DIR / "data" / "evaluation.json")
    report = evaluate_retrieval(cases, knowledge_base.list_chunks())
    print(json.dumps(report.model_dump(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Incident RAG POC")
    parser.add_argument(
        "command",
        choices=("serve", "evaluate"),
        nargs="?",
        default="serve",
        help="Start the API or evaluate sample retrieval cases.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluation()
        return
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
