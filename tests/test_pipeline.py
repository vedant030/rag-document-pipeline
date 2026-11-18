import json
from pathlib import Path

import numpy as np
import pytest

from rag_pipeline.chunking import chunk_text
from rag_pipeline.embeddings import HashingEmbedder
from rag_pipeline.lambda_handler import handler
from rag_pipeline.pipeline import ingest_document
from rag_pipeline.validation import (
    REASON_DUP_BATCH, REASON_DUP_INDEX, REASON_EMPTY, REASON_TOO_SHORT, validate_chunks,
)
from rag_pipeline.vectorstore import LocalVectorStore

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def emb():
    return HashingEmbedder()


# ---------- chunking ----------
def test_chunk_empty_and_whitespace_give_nothing():
    assert chunk_text("") == [] and chunk_text("  \n\n ") == []


def test_chunks_respect_size_and_keep_all_words():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Paragraph {i}" in joined for i in range(10))


def test_chunk_handles_text_with_no_separators():
    chunks = chunk_text("x" * 1000, size=300, overlap=30)
    assert all(len(c) <= 300 for c in chunks) and sum(len(c) for c in chunks) >= 1000


def test_chunk_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


# ---------- validation ----------
def test_validation_rejects_empty_short_and_duplicates():
    chunks = ["", "   ", "tiny", "A real chunk of text here.", "a  REAL chunk of   text here.", "Another distinct chunk of text."]
    r = validate_chunks(chunks, min_chars=10)
    reasons = {x.index: x.reason for x in r.rejected}
    assert reasons == {0: REASON_EMPTY, 1: REASON_EMPTY, 2: REASON_TOO_SHORT, 4: REASON_DUP_BATCH}
    assert [c.index for c in r.accepted] == [3, 5]


def test_validation_flags_chunks_already_in_index():
    first = validate_chunks(["Some chunk of reasonable length."])
    known = {c.content_hash for c in first.accepted}
    again = validate_chunks(["SOME CHUNK of reasonable length."], known_hashes=known)
    assert again.accepted == [] and again.rejected[0].reason == REASON_DUP_INDEX


# ---------- embeddings / store ----------
def test_embeddings_are_deterministic_and_normalised(emb):
    a, b = emb.embed(["customer data retention"]), emb.embed(["customer data retention"])
    assert np.array_equal(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_store_persists_to_disk(tmp_path, emb):
    store = LocalVectorStore(tmp_path / "db")
    ingest_document("a.md", "Backups are kept for thirty five days before expiry.", store, emb)
    reloaded = LocalVectorStore(tmp_path / "db")
    assert reloaded.count() == store.count() == 1
    hit = reloaded.search(emb.embed(["how long are backups kept"])[0], 1)[0]
    assert "Backups" in hit.text


# ---------- pipeline ----------
def test_unchanged_document_is_skipped_and_changed_one_replaces_old(emb):
    store = LocalVectorStore()
    assert ingest_document("a.md", "First version of the document text.", store, emb).status == "indexed"
    assert ingest_document("a.md", "First version of the document text.", store, emb).status == "unchanged"
    ingest_document("a.md", "Second version with entirely new wording here.", store, emb)
    assert store.count() == 1
    assert "Second" in store.search(emb.embed(["second version"])[0], 1)[0].text


def test_duplicate_document_under_another_name_is_not_indexed_twice(emb):
    store = LocalVectorStore()
    ingest_document("a.md", "Shared paragraph that appears in two files.", store, emb)
    report = ingest_document("b.md", "SHARED paragraph that appears in two files.", store, emb)
    assert report.status == "all_chunks_rejected" and report.rejected == {REASON_DUP_INDEX: 1}
    assert store.count() == 1


def test_empty_and_unsupported_documents(emb):
    store = LocalVectorStore()
    assert ingest_document("e.txt", "  \n ", store, emb).status == "empty"
    assert ingest_document("x.exe", b"MZ...", store, emb).status == "unsupported"
    assert store.count() == 0


def test_sample_corpus_end_to_end(emb):
    store = LocalVectorStore()
    reports = {p.name: ingest_document(p.name, p.read_bytes(), store, emb) for p in sorted(SAMPLES.iterdir())}
    assert reports["empty_notes.txt"].status == "empty"
    assert reports["retention_policy_copy.txt"].status == "all_chunks_rejected"
    assert store.count() == sum(r.indexed for r in reports.values()) >= 8

    def top(q):
        return store.search(emb.embed([q])[0], 1)[0].metadata["source"]

    assert top("how long do you keep customer data after an account is closed") == "data_retention_policy.md"
    assert top("what happens if a payment fails") == "billing_faq.md"
    assert top("who is the incident commander for a severity 1 outage") == "incident_response_runbook.md"
    assert top("how do I create an API key") == "onboarding_guide.md"


# ---------- lambda handler ----------
def _event(name, key, bucket="docs"):
    return {"Records": [{"eventName": name, "s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


def test_lambda_indexes_new_object_and_decodes_key(emb):
    store, objects = LocalVectorStore(), {("docs", "my notes/a b.md"): b"Lambda indexed document body text."}
    out = handler(_event("ObjectCreated:Put", "my+notes/a+b.md"), fetch=lambda b, k: objects[(b, k)], store=store, embedder=emb)
    assert out["processed"][0]["source"] == "s3://docs/my notes/a b.md"
    assert out["processed"][0]["indexed"] == 1 and store.count() == 1


def test_lambda_removes_chunks_on_delete(emb):
    store = LocalVectorStore()
    ingest_document("s3://docs/a.md", "Document that will be deleted soon.", store, emb)
    out = handler(_event("ObjectRemoved:Delete", "a.md"), fetch=None, store=store, embedder=emb)
    assert out["processed"][0] == {"source": "s3://docs/a.md", "status": "deleted", "chunks_removed": 1}
    assert store.count() == 0


def test_lambda_processes_other_records_then_raises_on_failure(emb):
    store = LocalVectorStore()
    ev = {"Records": _event("ObjectCreated:Put", "bad.md")["Records"] + _event("ObjectCreated:Put", "good.md")["Records"]}

    def fetch(bucket, key):
        if key == "bad.md":
            raise IOError("boom")
        return b"The good document still gets indexed."

    with pytest.raises(RuntimeError, match="bad.md"):
        handler(ev, fetch=fetch, store=store, embedder=emb)
    assert store.count() == 1
