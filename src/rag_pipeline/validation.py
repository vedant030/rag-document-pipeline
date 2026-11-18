"""Chunk validation: reject empty, too-short and duplicate chunks before indexing."""
import hashlib
from dataclasses import dataclass, field

REASON_EMPTY = "empty"
REASON_TOO_SHORT = "too_short"
REASON_DUP_BATCH = "duplicate_in_batch"
REASON_DUP_INDEX = "duplicate_in_index"


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive form used to detect duplicates."""
    return " ".join(text.lower().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    index: int
    text: str
    content_hash: str


@dataclass
class Rejection:
    index: int
    reason: str
    preview: str


@dataclass
class ValidationResult:
    accepted: list[Chunk] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejected:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return counts


def validate_chunks(chunks: list[str], known_hashes: set[str] | frozenset[str] = frozenset(), min_chars: int = 20) -> ValidationResult:
    """Split `chunks` into accepted and rejected.

    known_hashes: content hashes already in the index (from other documents).
    """
    result = ValidationResult()
    seen: set[str] = set()
    for i, text in enumerate(chunks):
        preview = text.strip()[:60]
        norm = normalize(text or "")
        if not norm:
            result.rejected.append(Rejection(i, REASON_EMPTY, preview))
            continue
        if len(norm) < min_chars:
            result.rejected.append(Rejection(i, REASON_TOO_SHORT, preview))
            continue
        h = content_hash(text)
        if h in known_hashes:
            result.rejected.append(Rejection(i, REASON_DUP_INDEX, preview))
            continue
        if h in seen:
            result.rejected.append(Rejection(i, REASON_DUP_BATCH, preview))
            continue
        seen.add(h)
        result.accepted.append(Chunk(i, text, h))
    return result
