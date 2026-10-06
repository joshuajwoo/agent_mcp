"""Dataset profiles and conversion to cited, retrieval-ready source documents."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from datasets import load_dataset

from rag_ingestion.chunking import chunk_text


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    context: str
    source_split: str
    dataset: str = "unknown"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexDocument:
    point_id: str
    text: str
    payload: dict[str, str | int]


@dataclass(frozen=True)
class DatasetProfile:
    """A source-specific, explicitly attributed ingestion profile."""

    key: str
    hub_id: str
    config: str | None
    text_field: str
    title_field: str | None
    fallback_title: str
    domain: str
    collection_kind: str
    perspective: str
    license: str


DATASET_PROFILES = {
    "philosophy": DatasetProfile(
        key="philosophy",
        hub_id="LisaMegaWatts/philosophy-corpus",
        config=None,
        text_field="text",
        title_field=None,
        fallback_title="Philosophy & Humanities Corpus",
        domain="philosophy",
        collection_kind="primary_and_reference_texts",
        perspective="historical works; not a normative consensus",
        license="MIT",
    ),
    "ethics": DatasetProfile(
        key="ethics",
        hub_id="hendrycks/ethics",
        config="commonsense",
        text_field="input",
        title_field=None,
        fallback_title="ETHICS Commonsense Scenario",
        domain="ethics",
        collection_kind="annotated_evaluation_scenarios",
        perspective="crowd-annotated moral acceptability; not moral authority",
        license="MIT",
    ),
    "theology": DatasetProfile(
        key="theology",
        hub_id="OpenChristianDataOrg/open-christian-data",
        config="structured_text",
        text_field="text",
        title_field="_source_title",
        fallback_title="Open Christian Data source text",
        domain="theology",
        collection_kind="historical_religious_texts",
        perspective="historical Christian sources; attributed, not universal theology",
        license="CC0-1.0 (dataset; inspect row-level source metadata)",
    ),
}


def _document_id(dataset: str, title: str, context: str) -> str:
    digest = hashlib.sha256(f"{dataset}\0{title}\0{context}".encode()).hexdigest()
    return f"{dataset}-{digest[:24]}"


def _point_id(document_id: str, chunk_index: int) -> str:
    """Return a stable UUID because Qdrant point IDs must be UUIDs or integers."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://mcp-rag.local/{document_id}/{chunk_index}"))


def _profile_metadata(profile: DatasetProfile, row: Mapping[str, object]) -> dict[str, str]:
    metadata = {
        "domain": profile.domain,
        "collection_kind": profile.collection_kind,
        "perspective": profile.perspective,
        "source_license": profile.license,
        "source_hub": profile.hub_id,
    }
    # Theology rows supply these values when known. Preserve them so retrieved
    # passages remain attributable and their representational limits are visible.
    for output_name, possible_fields in {
        "author": ("author", "_author"),
        "tradition": ("tradition", "_tradition"),
        "era": ("era", "_era"),
        "source_id": ("_source_id", "source_id"),
    }.items():
        value = next((row[name] for name in possible_fields if row.get(name)), None)
        if value is not None:
            metadata[output_name] = str(value)
    return metadata


def load_profile_documents(
    profile_key: str, split: str, max_contexts: int | None = None
) -> list[SourceDocument]:
    """Load one profile without treating its viewpoint as neutral fact."""
    try:
        profile = DATASET_PROFILES[profile_key]
    except KeyError as error:
        choices = ", ".join(sorted(DATASET_PROFILES))
        message = f"unknown dataset profile {profile_key!r}; choose one of {choices}"
        raise ValueError(message) from error
    if split != "train":
        raise ValueError(f"{profile_key} currently exposes only the train split")

    load_kwargs: dict[str, object] = {"split": split}
    if profile.config:
        rows = load_dataset(profile.hub_id, profile.config, **load_kwargs)
    else:
        rows = load_dataset(profile.hub_id, **load_kwargs)
    documents: list[SourceDocument] = []
    seen: set[str] = set()
    for row in rows:
        context = str(row.get(profile.text_field, "")).strip()
        if not context:
            continue
        title = str(row.get(profile.title_field, "")).strip() if profile.title_field else ""
        title = title or profile.fallback_title
        document_id = _document_id(profile.key, title, context)
        if document_id in seen:
            continue
        seen.add(document_id)
        documents.append(
            SourceDocument(
                document_id=document_id,
                title=title,
                context=context,
                source_split=split,
                dataset=profile.key,
                metadata=_profile_metadata(profile, row),
            )
        )
        if max_contexts is not None and len(documents) >= max_contexts:
            break
    return documents


def load_documents(
    dataset: str, split: str, max_contexts: int | None = None
) -> list[SourceDocument]:
    """Load one profile, or all profiles with the limit applied to each profile."""
    if dataset == "all":
        documents: list[SourceDocument] = []
        for profile_key in DATASET_PROFILES:
            documents.extend(load_profile_documents(profile_key, split, max_contexts))
        return documents
    return load_profile_documents(dataset, split, max_contexts)


def build_index_documents(
    source_documents: list[SourceDocument], *, max_words: int = 220, overlap_words: int = 40
) -> Iterator[IndexDocument]:
    """Create deterministic chunk records suitable for idempotent upserts."""
    for source in source_documents:
        for chunk_index, chunk in enumerate(
            chunk_text(source.context, max_words=max_words, overlap_words=overlap_words)
        ):
            chunk_id = f"{source.document_id}-chunk-{chunk_index}"
            payload: dict[str, str | int] = {
                "dataset": source.dataset,
                "source_split": source.source_split,
                "document_id": source.document_id,
                "chunk_id": chunk_id,
                "title": source.title,
                "chunk_index": chunk_index,
                "chunk_start_char": chunk.start_char,
                "chunk_end_char": chunk.end_char,
                "text": chunk.text,
            }
            payload.update(source.metadata)
            yield IndexDocument(
                point_id=_point_id(source.document_id, chunk_index),
                text=chunk.text,
                payload=payload,
            )
