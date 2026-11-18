"""AWS Lambda entry point, triggered by S3 object events.

Handler: rag_pipeline.lambda_handler.handler

* ObjectCreated:*  -> fetch the object, chunk, validate, embed, (re)index it.
* ObjectRemoved:*  -> delete that document's chunks from the index.

If any record fails, the handler raises after processing the rest, so Lambda's retry /
dead-letter-queue behaviour applies. Use a shared vector DB (VECTOR_STORE=chroma): the
local store lives in /tmp and is lost between cold starts.
"""
import json
import logging
from typing import Callable
from urllib.parse import unquote_plus

from .config import build_embedder, build_store
from .pipeline import ingest_document

log = logging.getLogger()
log.setLevel(logging.INFO)

_deps: dict = {}


def s3_fetch(bucket: str, key: str) -> bytes:
    import boto3

    return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


def _get_deps():
    if not _deps:
        _deps["store"] = build_store()
        _deps["embedder"] = build_embedder()
    return _deps["store"], _deps["embedder"]


def handler(event: dict, context=None, *, fetch: Callable[[str, str], bytes] | None = None, store=None, embedder=None) -> dict:
    """`fetch`, `store` and `embedder` can be injected for tests; defaults come from env."""
    if store is None or embedder is None:
        store, embedder = _get_deps()
    fetch = fetch or s3_fetch

    results, errors = [], []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        source = f"s3://{bucket}/{key}"
        try:
            if record.get("eventName", "").startswith("ObjectRemoved"):
                removed = store.delete_source(source)
                results.append({"source": source, "status": "deleted", "chunks_removed": removed})
            else:
                report = ingest_document(source, fetch(bucket, key), store, embedder)
                results.append(report.as_dict())
            log.info(json.dumps(results[-1]))
        except Exception as exc:  # keep going; fail the invocation at the end
            log.exception("failed processing %s", source)
            errors.append(f"{source}: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return {"processed": results}
