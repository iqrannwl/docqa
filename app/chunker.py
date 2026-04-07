"""
Text chunking — splits documents into overlapping chunks for embedding.
Uses a simple character-based splitter with sentence-boundary awareness.
"""

import re
from app.config import get_settings

settings = get_settings()


def split_into_chunks(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into chunks of ~chunk_size characters with overlap.
    Tries to break at sentence boundaries where possible.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= chunk_size:
        return [text]

    # Split into sentences (rough heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If single sentence exceeds chunk_size, hard-split it
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                    chunks.append(sentence[i : i + chunk_size])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to each chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks
