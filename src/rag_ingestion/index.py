"""Command-line corpus-profile-to-Qdrant indexing workflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from rag_ingestion.config import Settings
from rag_ingestion.dataset import (
    DATASET_PROFILES,
    IndexDocument,
    build_index_documents,
    load_documents,
)

LOGGER = logging.getLogger(__name__)


def batched(items: list[IndexDocument], size: int) -> Iterator[list[IndexDocument]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a cosine collection or reject an incompatible existing collection."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        LOGGER.info(
            "Created collection %s (%s dimensions, cosine distance)", collection_name, vector_size
        )
    else:
        collection = client.get_collection(collection_name)
        vector_config = collection.config.params.vectors
        if not isinstance(vector_config, models.VectorParams) or vector_config.size != vector_size:
            raise ValueError(
                f"Collection {collection_name!r} has a vector size incompatible "
                f"with {vector_size}. "
                "Set QDRANT_COLLECTION to a new name rather than overwriting it."
            )
    for field_name in ("dataset", "domain", "tradition", "title", "source_split"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


def index_documents(
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    documents: list[IndexDocument],
    batch_size: int,
) -> int:
    """Embed and upsert fixed batches. Stable IDs make reruns idempotent."""
    for batch_number, batch in enumerate(batched(documents, batch_size), start=1):
        vectors = embedder.encode(
            [document.text for document in batch], batch_size=batch_size, show_progress_bar=False
        ).tolist()
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(id=document.point_id, vector=vector, payload=document.payload)
                for document, vector in zip(batch, vectors, strict=True)
            ],
            wait=True,
        )
        LOGGER.info("Upserted batch %s (%s chunks)", batch_number, len(batch))
    return len(documents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index attributed text-corpus profiles in Qdrant Cloud."
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASET_PROFILES, "all"),
        default="all",
        help="Corpus profile to index. 'all' applies --max-contexts independently to each profile.",
    )
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-contexts", type=int, help="Limit unique contexts for a smoke test.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and chunk without loading a model or writing Qdrant.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args()
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("--max-contexts must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    sources = load_documents(args.dataset, args.split, args.max_contexts)
    documents = list(
        build_index_documents(
            sources, max_words=args.max_words, overlap_words=args.overlap_words
        )
    )
    LOGGER.info(
        "Prepared %s chunks from %s unique %s %s contexts",
        len(documents),
        len(sources),
        args.dataset,
        args.split,
    )
    if args.dry_run:
        return

    settings = Settings.from_environment()
    settings.validate_for_indexing()
    embedder = SentenceTransformer(settings.embedding_model)
    vector_size = embedder.get_embedding_dimension()
    if vector_size is None:
        raise RuntimeError("Embedding model did not expose its vector dimension")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_collection(client, settings.collection_name, vector_size)
    count = index_documents(client, settings.collection_name, embedder, documents, args.batch_size)
    LOGGER.info("Finished indexing %s chunks into %s", count, settings.collection_name)


if __name__ == "__main__":
    main()
