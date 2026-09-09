"""Qdrant-backed retrieval used by the MCP server."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from rag_ingestion.config import Settings


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieval result limited to fields useful to an MCP client."""

    chunk_id: str
    document_id: str
    title: str
    source_split: str
    text: str
    score: float
    chunk_start_char: int
    chunk_end_char: int

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "source_split": self.source_split,
            "text": self.text,
            "score": self.score,
            "chunk_start_char": self.chunk_start_char,
            "chunk_end_char": self.chunk_end_char,
        }


def _metadata_filter(title: str | None, source_split: str | None) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if title:
        conditions.append(models.FieldCondition(key="title", match=models.MatchValue(value=title)))
    if source_split:
        conditions.append(
            models.FieldCondition(key="source_split", match=models.MatchValue(value=source_split))
        )
    return models.Filter(must=conditions) if conditions else None


class QdrantRetriever:
    """Embed queries locally and retrieve their nearest Qdrant chunks."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedder = embedder

    @classmethod
    def from_environment(cls) -> QdrantRetriever:
        settings = Settings.from_environment()
        settings.validate_for_indexing()
        return cls(
            client=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
            collection_name=settings.collection_name,
            embedder=SentenceTransformer(settings.embedding_model),
        )

    def search(self,
        query: str,
        top_k: int,
        *,
        title: str | None = None,
        source_split: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        vector = self._embedder.encode([query], show_progress_bar=False)[0].tolist()
        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=_metadata_filter(title, source_split),
            with_payload=True,
            limit=top_k,
        ).points
        return [_result_from_payload(hit.payload, float(hit.score)) for hit in hits]

    def filter_by_metadata(
        self, *, title: str | None, source_split: str | None, limit: int
    ) -> list[RetrievalResult]:
        if not title and not source_split:
            raise ValueError("provide title or source_split")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=_metadata_filter(title, source_split),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [_result_from_payload(point.payload, score=0.0) for point in points]


def _result_from_payload(payload: dict[str, object] | None, score: float) -> RetrievalResult:
    if not payload:
        raise ValueError("Qdrant point did not include a payload")
    required = (
        "chunk_id",
        "document_id",
        "title",
        "source_split",
        "text",
        "chunk_start_char",
        "chunk_end_char",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Qdrant point payload is missing fields: {', '.join(missing)}")
    return RetrievalResult(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        title=str(payload["title"]),
        source_split=str(payload["source_split"]),
        text=str(payload["text"]),
        score=score,
        chunk_start_char=int(payload["chunk_start_char"]),
        chunk_end_char=int(payload["chunk_end_char"]),
    )
