"""Embedders.

HashingEmbedder is deterministic, needs no model download and no API key, so the whole
pipeline runs offline and in tests. It is a lexical (bag-of-words) embedding: good enough
to demo retrieval, not a substitute for a semantic model. Use SentenceTransformerEmbedder
(or any object with `embed(list[str]) -> np.ndarray`) for semantic search.
"""
import hashlib
import math
import re
from typing import Protocol

import numpy as np

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was we what when which with you your do does can if my me our us they them their there then than so not but will would should could may might also any all about into over after before up out no yes".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for tok in _TOKEN.findall(text.lower()):
                if tok not in _STOPWORDS:
                    counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                idx, sign = self._bucket(tok)
                out[row, idx] += sign * (1.0 + math.log(c))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    """Semantic embeddings (optional dependency: sentence-transformers)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
