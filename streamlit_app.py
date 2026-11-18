"""Interactive demo: ingest the sample documents, inspect validation results, run searches.

Run: streamlit run streamlit_app.py
Runs fully offline (hashing embedder + in-memory store); upload your own .txt/.md files too.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_pipeline.embeddings import HashingEmbedder  # noqa: E402
from rag_pipeline.pipeline import ingest_document  # noqa: E402
from rag_pipeline.vectorstore import LocalVectorStore  # noqa: E402

st.set_page_config(page_title="RAG Document Pipeline", layout="wide")
st.title("RAG Document Pipeline")
st.caption(
    "Chunk, validate, embed and index documents, then search them. Demo mode: in-memory vector "
    "store and a lexical hashing embedder. The S3/Lambda/Airflow parts are in the GitHub repo."
)

embedder = HashingEmbedder()


@st.cache_resource
def sample_index():
    store, reports = LocalVectorStore(), []
    for path in sorted((ROOT / "sample_docs").iterdir()):
        reports.append(ingest_document(path.name, path.read_bytes(), store, embedder))
    return store, reports


store, reports = sample_index()

with st.sidebar:
    st.header("Add your own documents")
    uploads = st.file_uploader("Upload .txt or .md", type=["txt", "md"], accept_multiple_files=True)
    extra = []
    if uploads:
        # Each rerun re-ingests; unchanged files are skipped by content hash.
        for up in uploads:
            extra.append(ingest_document(up.name, up.getvalue(), store, embedder))

st.subheader("Ingestion report")
st.write(
    "Empty chunks, near-empty chunks and duplicates are rejected before indexing. "
    "`retention_policy_copy.txt` is an upper-cased copy of another file, and `empty_notes.txt` is blank."
)
st.dataframe(
    [
        {
            "document": r.source,
            "status": r.status,
            "chunks": r.total_chunks,
            "indexed": r.indexed,
            "rejected": ", ".join(f"{k}: {v}" for k, v in r.rejected.items()) or "-",
        }
        for r in reports + extra
    ],
    use_container_width=True,
)
st.metric("Chunks in index", store.count())

st.subheader("Search")
query = st.text_input("Ask something", "How long do you keep customer data after an account is closed?")
if query:
    for hit in store.search(embedder.embed([query])[0], 3):
        with st.container(border=True):
            st.markdown(f"**{hit.metadata['source']}** · similarity {hit.score:.2f}")
            st.write(hit.text)
