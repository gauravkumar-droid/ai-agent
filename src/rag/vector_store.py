from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .documents import Document


@dataclass
class SearchResult:
    document: Document
    score: float


class VectorStore:
    """
    Very small TF-IDF based vector store for local retrieval.
    """

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=4096,
            norm="l2",
        )
        self._matrix = None
        self._documents: List[Document] = []

    def fit(self, documents: Sequence[Document]) -> None:
        texts = [doc.text for doc in documents]
        self._matrix = self.vectorizer.fit_transform(texts)
        self._documents = list(documents)

    def is_fit(self) -> bool:
        return self._matrix is not None and bool(self._documents)

    def similarity_search(self, query: str, k: int = 5) -> List[SearchResult]:
        if not self.is_fit():
            raise RuntimeError("Vector store is empty. Run ingest first.")
        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]
        return [SearchResult(self._documents[idx], float(scores[idx])) for idx in top_indices]

    def save(self, path: Path | str) -> None:
        if not self.is_fit():
            raise RuntimeError("Cannot save an empty vector store.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "matrix": self._matrix,
            "documents": self._documents,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "VectorStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        payload = joblib.load(path)
        store = cls(vectorizer=payload["vectorizer"])
        store._matrix = payload["matrix"]
        store._documents = payload["documents"]
        return store
