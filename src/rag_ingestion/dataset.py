"""SQuAD loading and conversion to deduplicated source documents."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from rag_ingestion.chunking import chunk_text

SQUAD_DATASET = "rajpurkar/squad"


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    context: str
    source_split: str


@dataclass(frozen=True)
class IndexDocument:
    point_id: str
    text: str
    payload: dict[str, str | int]


def _document_id(title: str, context: str) -> str:
    digest = hashlib.sha256(f"{title}\0{context}".encode()).hexdigest()
    return f"squad-{digest[:24]}"


def _point_id(document_id: str, chunk_index: int) -> str:
    """Return a stable UUID because Qdrant point IDs must be UUIDs or integers."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://mcp-rag.local/{document_id}/{chunk_index}"))


def load_squad_contexts(split: str, max_contexts: int | None = None) -> list[SourceDocument]:
    """Load a split and de-duplicate contexts repeated for every question row."""
    if split not in {"train", "validation"}:
        raise ValueError("SQuAD supports only the train or validation split")
    rows = load_dataset(SQUAD_DATASET, split=split)
    seen: set[str] = set()
    documents: list[SourceDocument] = []
    for row in rows:
        title, context = row["title"], row["context"]
        document_id = _document_id(title, context)
        if document_id in seen:
            continue
        seen.add(document_id)
        documents.append(SourceDocument(document_id, title, context, split))
        if max_contexts is not None and len(documents) >= max_contexts:
            break
    return documents


def build_index_documents(
    source_documents: list[SourceDocument], *, max_words: int = 220, overlap_words: int = 40
) -> Iterator[IndexDocument]:
    """Create deterministic chunk records suitable for idempotent upserts."""
    for source in source_documents:
        for chunk_index, chunk in enumerate(
            chunk_text(source.context, max_words=max_words, overlap_words=overlap_words)
        ):
            chunk_id = f"{source.document_id}-chunk-{chunk_index}"
            yield IndexDocument(
                point_id=_point_id(source.document_id, chunk_index),
                text=chunk.text,
                payload={
                    "dataset": "squad_v1",
                    "source_split": source.source_split,
                    "document_id": source.document_id,
                    "chunk_id": chunk_id,
                    "title": source.title,
                    "chunk_index": chunk_index,
                    "chunk_start_char": chunk.start_char,
                    "chunk_end_char": chunk.end_char,
                    "text": chunk.text,
                },
            )
