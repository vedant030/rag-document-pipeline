# RAG Document Pipeline with Data Quality Checks

Documents dropped into S3 are automatically chunked, validated, embedded and indexed into a vector database. Airflow re-scans the bucket on a schedule to keep the index fresh. Validation rejects empty, too-short and duplicate chunks **before** they reach the index.

```
S3 upload / delete ──> Lambda ──┐
                                ├─> load -> chunk -> VALIDATE -> embed -> vector DB
Airflow (daily rescan) ─────────┘                       │
                                                        └─> rejected: empty | too_short | duplicate_in_batch | duplicate_in_index
```

## Try it in 30 seconds (no AWS, no API keys)

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
pip install -e .
python -m rag_pipeline ingest sample_docs
python -m rag_pipeline query "how long do you keep customer data after an account is closed?"
streamlit run streamlit_app.py     # interactive demo
```

`sample_docs/` includes an upper-cased copy of another file and a blank file, so you can see the validation reject them.

## Data quality checks

| Check | Reason code | Example |
|---|---|---|
| Empty chunk | `empty` | whitespace-only text |
| Too short | `too_short` | fewer than 20 characters after normalising |
| Duplicate within a document | `duplicate_in_batch` | a paragraph repeated twice |
| Duplicate of something already indexed | `duplicate_in_index` | same text in another file |

Duplicates are detected on a **normalised hash** (lower-cased, whitespace collapsed), so `"Hello  World"` and `"hello world"` match. Every ingest returns a report with the counts per reason.

Ingestion is **idempotent**: unchanged documents are skipped by content hash, and a changed document replaces its old chunks, so re-running the Airflow DAG never creates duplicates.

## Layout

| Path | Purpose |
|---|---|
| `src/rag_pipeline/chunking.py` | Recursive text splitter (built-in, or LangChain with `CHUNKER=langchain`) |
| `src/rag_pipeline/validation.py` | The data quality checks |
| `src/rag_pipeline/embeddings.py` | Embedder interface; hashing (offline) and sentence-transformers |
| `src/rag_pipeline/vectorstore.py` | Store interface; local numpy store and Chroma adapter |
| `src/rag_pipeline/pipeline.py` | `ingest_document`: the whole flow for one document |
| `src/rag_pipeline/lambda_handler.py` | S3-event Lambda (create/update and delete) |
| `airflow/dags/rag_refresh_dag.py` | Daily S3 rescan and refresh |
| `infra/template.yaml` | AWS SAM template: bucket, Lambda, S3 notifications |
| `streamlit_app.py` | Interactive demo |

## Deploying on AWS

1. Run a Chroma server reachable from your Lambda (or swap in another vector DB by implementing the small `VectorStore` interface).
2. `sam build && sam deploy --guided` using `infra/template.yaml` (pass your Chroma host).
3. Upload files under `documents/` in the bucket; the Lambda indexes them. Deleting a file removes its chunks.
4. Copy `airflow/` and `src/` to your Airflow environment, set `DOCS_BUCKET`, and enable `rag_index_refresh`.

Environment variables: `EMBEDDER` (`hashing` | `sentence-transformers`), `VECTOR_STORE` (`local` | `chroma`), `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`, `CHUNKER` (`builtin` | `langchain`).
