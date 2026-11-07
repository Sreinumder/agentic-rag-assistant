from __future__ import annotations

from pathlib import Path

from src.config import Config


class Document:
    def __init__(self, content: str, metadata: dict | None = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"Document(length={len(self.content)}, preview='{preview}...')"


SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml", ".toml"}


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(source: str | Path) -> list[Document]:
    source = Path(source)
    documents: list[Document] = []

    if source.is_file():
        content = _read_file(source)
        documents.append(Document(content=content, metadata={"source": str(source)}))
    elif source.is_dir():
        for file_path in sorted(source.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                content = _read_file(file_path)
                documents.append(Document(content=content, metadata={"source": str(file_path)}))
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    # Imported lazily: file loading and Document construction stay usable in
    # environments where the optional chunking dependency is absent.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_text(doc.content)
        for i, split in enumerate(splits):
            metadata = {**doc.metadata, "chunk_index": i}
            chunks.append(Document(content=split, metadata=metadata))

    return chunks
