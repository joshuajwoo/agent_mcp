import uuid

from rag_ingestion.dataset import SourceDocument, build_index_documents


def test_index_documents_have_deterministic_ids_and_retrieval_payload() -> None:
    source = SourceDocument(
        document_id="squad-example",
        title="Example title",
        context="First sentence. Second sentence.",
        source_split="train",
    )

    documents = list(build_index_documents([source], max_words=10, overlap_words=0))

    assert uuid.UUID(documents[0].point_id).version == 5
    assert documents[0].payload["chunk_id"] == "squad-example-chunk-0"
    assert documents[0].payload["dataset"] == "squad_v1"
    assert documents[0].payload["title"] == "Example title"
