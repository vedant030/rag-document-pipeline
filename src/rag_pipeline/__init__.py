"""RAG document pipeline: load -> chunk -> validate -> embed -> index."""
from .pipeline import IngestReport, ingest_document

__all__ = ["IngestReport", "ingest_document"]
