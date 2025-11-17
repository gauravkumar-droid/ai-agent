from __future__ import annotations

import argparse
import json

from .rag_engine import RAGEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG over API enablement specs")
    parser.add_argument(
        "--spec-dir",
        default="data/api_specs",
        help="Directory containing API spec YAML files",
    )
    parser.add_argument(
        "--store-path",
        default="artifacts/vector_store.joblib",
        help="Path to persist the vector store",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Ingest YAML specs into the vector store")
    ingest_parser.set_defaults(func=run_ingest)

    query_parser = sub.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="Natural language question to answer")
    query_parser.add_argument("--k", type=int, default=4, help="Number of contexts to retrieve")
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Return machine-readable JSON output",
    )
    query_parser.add_argument(
        "--api-name",
        help="Restrict retrieval to a specific API/application (e.g., 'Zendesk Support API')",
    )
    query_parser.set_defaults(func=run_query)

    return parser.parse_args()


def run_ingest(args: argparse.Namespace) -> None:
    engine = RAGEngine(spec_dir=args.spec_dir, store_path=args.store_path)
    count = engine.ingest()
    print(f"Ingested {count} documents from {args.spec_dir} -> {args.store_path}")


def run_query(args: argparse.Namespace) -> None:
    engine = RAGEngine(spec_dir=args.spec_dir, store_path=args.store_path)
    result = engine.query(args.question, k=args.k, api_name=args.api_name)
    if args.json:
        payload = {
            "answer": result.answer,
            "contexts": [ctx.__dict__ for ctx in result.contexts],
            "api_filter": result.api_filter,
        }
        print(json.dumps(payload, indent=2))
        return

    print(result.answer)
    print("\nTop contexts:")
    for ctx in result.contexts:
        location = ctx.api_name or "Unknown API"
        if ctx.operation:
            location += f" — {ctx.operation}"
        print(f"- {location} [{ctx.section}] score={ctx.score:.3f}")
        print(f"  {ctx.excerpt}\n")


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
