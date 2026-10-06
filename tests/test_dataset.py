import uuid

import rag_ingestion.dataset as dataset_module
from rag_ingestion.dataset import SourceDocument, build_index_documents, load_profile_documents


def test_index_documents_have_deterministic_ids_and_retrieval_payload() -> None:
    source = SourceDocument(
        document_id="philosophy-example",
        title="Example title",
        context="First sentence. Second sentence.",
        source_split="train",
        dataset="philosophy",
        metadata={"domain": "philosophy"},
    )

    documents = list(build_index_documents([source], max_words=10, overlap_words=0))

    assert uuid.UUID(documents[0].point_id).version == 5
    assert documents[0].payload["chunk_id"] == "philosophy-example-chunk-0"
    assert documents[0].payload["dataset"] == "philosophy"
    assert documents[0].payload["domain"] == "philosophy"
    assert documents[0].payload["title"] == "Example title"


def test_profile_loader_preserves_provenance_and_limits_documents(monkeypatch) -> None:
    rows = [
        {"text": "A source passage.", "_source_title": "A Work", "_author": "An Author"},
        {"text": "A source passage.", "_source_title": "A Work"},
        {"text": "Another source passage.", "_source_title": "Another Work"},
    ]
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *_, **__: rows)

    documents = load_profile_documents("theology", "train", max_contexts=2)

    assert len(documents) == 2
    assert documents[0].dataset == "theology"
    assert documents[0].title == "A Work"
    assert documents[0].metadata["author"] == "An Author"
    assert documents[0].metadata["perspective"].startswith("historical Christian")
