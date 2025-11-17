from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .document_builder import DocumentBuilder
from .spec_loader import load_api_specs
from .vector_store import SearchResult, VectorStore


@dataclass
class RetrievedContext:
    api_name: str
    section: str
    operation: str | None
    score: float
    excerpt: str


@dataclass
class RAGAnswer:
    answer: str
    contexts: List[RetrievedContext]


class RAGEngine:
    """
    Coordinates ingestion and retrieval over API enablement knowledge.
    """

    def __init__(self, spec_dir: str | Path, store_path: str | Path):
        self.spec_dir = Path(spec_dir)
        self.store_path = Path(store_path)
        self._builder = DocumentBuilder()
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore.load(self.store_path)
        return self._store

    def ingest(self) -> int:
        specs = load_api_specs(self.spec_dir)
        documents = self._builder.build_corpus(specs)
        store = VectorStore()
        store.fit(documents)
        store.save(self.store_path)
        self._store = store
        return len(documents)

    def query(self, question: str, k: int = 4) -> RAGAnswer:
        results = self.store.similarity_search(question, k)
        contexts = [self._to_context(result) for result in results]
        synthesized = self._compose_answer(question, contexts)
        return RAGAnswer(answer=synthesized, contexts=contexts)

    def _to_context(self, result: SearchResult) -> RetrievedContext:
        metadata = result.document.metadata or {}
        excerpt = "\n".join(result.document.text.splitlines()[:12])
        return RetrievedContext(
            api_name=metadata.get("api_name"),
            section=metadata.get("section"),
            operation=metadata.get("operation"),
            score=result.score,
            excerpt=excerpt,
        )

    def _compose_answer(self, question: str, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return "No matching API knowledge found. Re-run ingestion or add more specs."

        lines = [
            f"Question: {question}",
            "Likely API implementations and requirements:",
        ]
        for ctx in contexts:
            location = ctx.api_name or "Unknown API"
            if ctx.operation:
                location += f" — {ctx.operation}"
            lines.append(f"- {location} [{ctx.section}] (score={ctx.score:.3f})")
            lines.append(f"  {ctx.excerpt.replace(chr(10), ' ')}")
        lines.append(
            "Cross-check authentication instructions and schema details above before triggering real requests."
        )
        return "\n".join(lines)
