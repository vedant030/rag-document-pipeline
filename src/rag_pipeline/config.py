"""Build the embedder and vector store from environment variables.

EMBEDDER      hashing (default) | sentence-transformers
VECTOR_STORE  local (default)   | chroma
VECTOR_PATH   directory for the local store (default: .rag_store)
CHROMA_HOST / CHROMA_PORT / CHROMA_COLLECTION   for VECTOR_STORE=chroma
"""
import os

from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .vectorstore import ChromaVectorStore, LocalVectorStore


def build_embedder(name: str | None = None):
    name = name or os.environ.get("EMBEDDER", "hashing")
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown EMBEDDER: {name}")


def build_store(kind: str | None = None):
    kind = kind or os.environ.get("VECTOR_STORE", "local")
    if kind == "local":
        return LocalVectorStore(os.environ.get("VECTOR_PATH", ".rag_store"))
    if kind == "chroma":
        return ChromaVectorStore(
            collection=os.environ.get("CHROMA_COLLECTION", "documents"),
            host=os.environ.get("CHROMA_HOST"),
            port=int(os.environ.get("CHROMA_PORT", "8000")),
            path=os.environ.get("VECTOR_PATH"),
        )
    raise ValueError(f"unknown VECTOR_STORE: {kind}")
