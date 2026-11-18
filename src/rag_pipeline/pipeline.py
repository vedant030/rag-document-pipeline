"""Core ingestion: load -> chunk -> validate -> embed -> index (idempotent per source)."""
import hashlib
from dataclasses import dataclass, field

from .chunking import chunk_text
from .embeddings import Embedder
from .loaders import UnsupportedFormat, load_text
from .validation import validate_chunks
from .vectorstore import VectorStore

STATUS_INDEXED = "indexed"
STATUS_UNCHANGED = "unchanged"
STATUS_EMPTY = "empty"
STATUS_ALL_REJECTED = "all_chunks_rejected"
STATUS_UNSUPPORTED = "unsupported"


@dataclass
class IngestReport:
    source: str
    status: str
    total_chunks: int = 0
    indexed: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "indexed": self.indexed,
            "rejected": self.rejected,
            "detail": self.detail,
        }


def ingest_document(
    source: str,
    data: bytes | str,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chars: int = 20,
) -> IngestReport:
    """Index one document. Safe to call repeatedly: unchanged documents are skipped and
    changed documents replace their previous chunks."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    try:
        text = load_text(source, raw)
    except UnsupportedFormat as exc:
        return IngestReport(source, STATUS_UNSUPPORTED, detail=str(exc))

    doc_hash = hashlib.sha256(raw).hexdigest()
    if store.source_hash(source) == doc_hash:
        return IngestReport(source, STATUS_UNCHANGED)

    store.delete_source(source)  # refresh: drop stale chunks first

    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return IngestReport(source, STATUS_EMPTY, detail="document has no text")

    result = validate_chunks(chunks, known_hashes=store.known_hashes(), min_chars=min_chars)
    if result.accepted:
        vectors = embedder.embed([c.text for c in result.accepted])
        store.add(
            ids=[f"{doc_hash[:12]}-{c.index}" for c in result.accepted],
            vectors=vectors,
            texts=[c.text for c in result.accepted],
            metadatas=[
                {"source": source, "chunk_index": c.index, "content_hash": c.content_hash, "doc_hash": doc_hash}
                for c in result.accepted
            ],
        )
    return IngestReport(
        source,
        STATUS_INDEXED if result.accepted else STATUS_ALL_REJECTED,
        total_chunks=len(chunks),
        indexed=len(result.accepted),
        rejected=result.reason_counts(),
    )
