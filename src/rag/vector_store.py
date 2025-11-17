from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
        self._api_index: Dict[str, List[int]] = {}
        self._api_aliases: Dict[str, set[str]] = {}

    def fit(self, documents: Sequence[Document]) -> None:
        texts = [doc.text for doc in documents]
        self._matrix = self.vectorizer.fit_transform(texts)
        self._documents = list(documents)
        self._rebuild_api_metadata()

    def is_fit(self) -> bool:
        return self._matrix is not None and bool(self._documents)

    def similarity_search(self, query: str, k: int = 5, api_name: str | None = None) -> List[SearchResult]:
        if not self.is_fit():
            raise RuntimeError("Vector store is empty. Run ingest first.")
        if api_name:
            subset_indices = self._api_index.get(api_name)
            if not subset_indices:
                raise ValueError(f"No documents found for API '{api_name}'.")
            matrix = self._matrix[subset_indices]
            docs = [self._documents[idx] for idx in subset_indices]
        else:
            matrix = self._matrix
            docs = self._documents

        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]
        return [SearchResult(docs[idx], float(scores[idx])) for idx in top_indices]

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
        store._rebuild_api_metadata()
        return store

    def get_api_names(self) -> List[str]:
        return sorted(self._api_index.keys())

    def get_api_aliases(self) -> Dict[str, List[str]]:
        return {name: sorted(aliases) for name, aliases in self._api_aliases.items()}

    def _rebuild_api_metadata(self) -> None:
        api_index: Dict[str, List[int]] = {}
        api_aliases: Dict[str, set[str]] = {}
        for idx, doc in enumerate(self._documents):
            api_name = doc.metadata.get("api_name")
            if not api_name:
                continue
            api_index.setdefault(api_name, []).append(idx)
            alias_set = api_aliases.setdefault(api_name, set())
            alias_set.add(api_name)
            application = doc.metadata.get("application")
            if application:
                alias_set.add(application)
            for alias in doc.metadata.get("aliases") or []:
                alias_set.add(alias)
        self._api_index = api_index
        self._api_aliases = api_aliases
