"""Text chunking.

The built-in recursive splitter has no dependencies. Set CHUNKER=langchain (or pass
backend="langchain") to use LangChain's RecursiveCharacterTextSplitter instead.
"""
import os

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Break text into pieces no longer than `size`, preferring natural boundaries.

    Each piece keeps the separator it was split on, so joining pieces restores the text.
    """
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    out: list[str] = []
    for piece in pieces:
        if not piece.strip():
            continue
        out.extend(_split(piece, size, rest) if len(piece) > size else [piece])
    return out


def _builtin_chunks(text: str, size: int, overlap: int) -> list[str]:
    pieces = _split(text.strip(), size, SEPARATORS)
    chunks: list[str] = []
    cur: list[str] = []

    def length(parts: list[str]) -> int:
        return sum(len(p) for p in parts)

    for piece in pieces:
        if cur and length(cur) + len(piece) > size:
            chunk = "".join(cur).strip()
            if chunk:
                chunks.append(chunk)
            # carry trailing pieces forward as overlap
            tail: list[str] = []
            for prev in reversed(cur):
                if length(tail) + len(prev) > overlap:
                    break
                tail.insert(0, prev)
            cur = tail
            while cur and length(cur) + len(piece) > size:
                cur.pop(0)
        cur.append(piece)
    chunk = "".join(cur).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def chunk_text(text: str, size: int = 500, overlap: int = 50, backend: str | None = None) -> list[str]:
    """Split text into chunks of at most ~`size` characters. Empty text gives []."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    if not text or not text.strip():
        return []
    backend = backend or os.environ.get("CHUNKER", "builtin")
    if backend == "langchain":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        return splitter.split_text(text)
    return _builtin_chunks(text, size, overlap)
