from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text generated from a longer document."""

    document_id: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    document_id: str,
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[TextChunk]:
    """
    Split a long document into overlapping text chunks.

    Args:
        document_id: Unique identifier for the source document.
        text: Raw document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of TextChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized_text = " ".join(text.split())
    chunks: List[TextChunk] = []

    start = 0
    chunk_id = 0

    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))

        # Prefer ending at a whitespace boundary when possible.
        if end < len(normalized_text):
            boundary = normalized_text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=chunk,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_id += 1

        if end >= len(normalized_text):
            break

        start = max(0, end - overlap)

    return chunks


def chunk_documents(
    documents: Iterable[Tuple[str, str]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[TextChunk]:
    """
    Chunk multiple documents.

    Args:
        documents: Iterable of (document_id, text) pairs.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A flat list of chunks across all documents.
    """
    all_chunks: List[TextChunk] = []

    for document_id, text in documents:
        all_chunks.extend(
            chunk_text(
                document_id=document_id,
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return all_chunks
