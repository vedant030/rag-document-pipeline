"""Command line: ingest a folder, then query it.

  python -m rag_pipeline ingest sample_docs
  python -m rag_pipeline query "how long do we keep customer data?"
"""
import argparse
import json
from pathlib import Path

from .config import build_embedder, build_store
from .pipeline import ingest_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index every document in a folder")
    ing.add_argument("folder")
    q = sub.add_parser("query", help="search the index")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)

    store, embedder = build_store(), build_embedder()

    if args.cmd == "ingest":
        for path in sorted(Path(args.folder).rglob("*")):
            if path.is_file():
                report = ingest_document(str(path), path.read_bytes(), store, embedder)
                print(json.dumps(report.as_dict()))
        print(f"index now holds {store.count()} chunks")
        return 0

    for hit in store.search(embedder.embed([args.text])[0], args.k):
        print(f"[{hit.score:.3f}] {hit.metadata['source']}\n    {hit.text[:200]!r}")
    return 0
