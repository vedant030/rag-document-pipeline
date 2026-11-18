"""Scheduled refresh: re-scan the S3 prefix and re-index anything new or changed.

Unchanged documents are skipped by content hash, so a full scan is cheap. This also
repairs the index if a Lambda invocation was missed.
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rag_pipeline.config import build_embedder, build_store  # noqa: E402
from rag_pipeline.lambda_handler import s3_fetch  # noqa: E402
from rag_pipeline.pipeline import ingest_document  # noqa: E402

BUCKET = os.environ.get("DOCS_BUCKET", "YOUR_BUCKET")
PREFIX = os.environ.get("DOCS_PREFIX", "documents/")
SUPPORTED = (".txt", ".md", ".markdown", ".rst", ".pdf")


def list_keys(bucket: str, prefix: str) -> list[str]:
    import boto3

    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    return [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].lower().endswith(SUPPORTED)
    ]


def refresh_index(**_):
    store, embedder = build_store(), build_embedder()
    reports, failures = [], []
    for key in list_keys(BUCKET, PREFIX):
        source = f"s3://{BUCKET}/{key}"
        try:
            reports.append(ingest_document(source, s3_fetch(BUCKET, key), store, embedder))
        except Exception as exc:
            failures.append(f"{source}: {exc}")
    summary = {
        "documents": len(reports),
        "indexed": sum(r.status == "indexed" for r in reports),
        "unchanged": sum(r.status == "unchanged" for r in reports),
        "chunks_added": sum(r.indexed for r in reports),
        "chunks_rejected": sum(sum(r.rejected.values()) for r in reports),
    }
    print(summary)
    if failures:
        raise RuntimeError("Refresh failures: " + "; ".join(failures))
    return summary


with DAG(
    dag_id="rag_index_refresh",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-eng", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["rag", "vector-db"],
) as dag:
    PythonOperator(task_id="refresh_index", python_callable=refresh_index)
