"""Turn raw object bytes into text, based on file extension."""
import io

TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".rst")


class UnsupportedFormat(Exception):
    pass


def load_text(key: str, data: bytes) -> str:
    name = key.lower()
    if name.endswith(TEXT_EXTENSIONS):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise UnsupportedFormat("PDF support needs `pip install pypdf`") from exc
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise UnsupportedFormat(f"unsupported file type: {key}")
