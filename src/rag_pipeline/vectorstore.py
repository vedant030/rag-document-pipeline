"""Vector stores behind one small interface.

LocalVectorStore: numpy cosine search with optional on-disk persistence (no dependencies).
ChromaVectorStore: adapter for a Chroma server / persistent client (optional dependency).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class SearchHit:
    text: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict]) -> None: ...
    def search(self, vector: np.ndarray, k: int = 4) -> list[SearchHit]: ...
    def known_hashes(self) -> set[str]: ...
    def source_hash(self, source: str) -> str | None: ...
    def delete_source(self, source: str) -> int: ...
    def count(self) -> int: ...


class LocalVectorStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        if self.path and (self.path / "records.json").exists():
            self._load()

    # -- persistence --
    def _load(self) -> None:
        rec = json.loads((self.path / "records.json").read_text())
        self._ids, self._texts, self._meta = rec["ids"], rec["texts"], rec["meta"]
        self._vectors = np.load(self.path / "vectors.npy")

    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "records.json").write_text(
            json.dumps({"ids": self._ids, "texts": self._texts, "meta": self._meta})
        )
        np.save(self.path / "vectors.npy", self._vectors)

    # -- interface --
    def add(self, ids, vectors, texts, metadatas) -> None:
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors.size == 0 else np.vstack([self._vectors, vectors])
        self._ids += list(ids)
        self._texts += list(texts)
        self._meta += list(metadatas)
        self._save()

    def search(self, vector, k: int = 4) -> list[SearchHit]:
        if not self._ids:
            return []
        scores = self._vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        top = np.argsort(-scores)[:k]
        return [SearchHit(self._texts[i], float(scores[i]), self._meta[i]) for i in top]

    def known_hashes(self) -> set[str]:
        return {m["content_hash"] for m in self._meta}

    def source_hash(self, source: str) -> str | None:
        for m in self._meta:
            if m.get("source") == source:
                return m.get("doc_hash")
        return None

    def delete_source(self, source: str) -> int:
        keep = [i for i, m in enumerate(self._meta) if m.get("source") != source]
        removed = len(self._ids) - len(keep)
        if removed:
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._save()
        return removed

    def count(self) -> int:
        return len(self._ids)


class ChromaVectorStore:
    """Chroma adapter. Pass host/port for a server, or path for a local persistent DB."""

    def __init__(self, collection: str = "documents", host: str | None = None, port: int = 8000, path: str | None = None):
        import chromadb

        client = chromadb.HttpClient(host=host, port=port) if host else chromadb.PersistentClient(path=path or ".chroma")
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids, vectors, texts, metadatas) -> None:
        if ids:
            self._col.add(ids=list(ids), embeddings=np.asarray(vectors).tolist(), documents=list(texts), metadatas=list(metadatas))

    def search(self, vector, k: int = 4) -> list[SearchHit]:
        res = self._col.query(query_embeddings=[np.asarray(vector).reshape(-1).tolist()], n_results=k)
        return [
            SearchHit(doc, 1.0 - dist, meta)  # cosine distance -> similarity
            for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
        ]

    def known_hashes(self) -> set[str]:
        return {m["content_hash"] for m in self._col.get(include=["metadatas"])["metadatas"]}

    def source_hash(self, source: str) -> str | None:
        res = self._col.get(where={"source": source}, limit=1, include=["metadatas"])
        return res["metadatas"][0].get("doc_hash") if res["metadatas"] else None

    def delete_source(self, source: str) -> int:
        existing = self._col.get(where={"source": source}, include=[])["ids"]
        if existing:
            self._col.delete(ids=existing)
        return len(existing)

    def count(self) -> int:
        return self._col.count()
