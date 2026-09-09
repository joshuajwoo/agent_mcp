"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for a Qdrant indexing run."""

    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "squad_v1_bge_small"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY", ""),
            collection_name=os.environ.get("QDRANT_COLLECTION", "squad_v1_bge_small"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )

    def validate_for_indexing(self) -> None:
        missing = [
            name
            for name, value in {
                "QDRANT_URL": self.qdrant_url,
                "QDRANT_API_KEY": self.qdrant_api_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Export these values before indexing."
            )
