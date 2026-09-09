import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_ingestion.dataset import SourceDocument, build_index_documents
from rag_ingestion.index import ensure_collection, index_documents

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


class FakeEmbedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.array([[float(len(text)), 1.0] for text in texts])


def test_indexing_upserts_qdrant_compatible_points() -> None:
    client = QdrantClient(":memory:")
    collection_name = "test_collection"
    documents = list(
        build_index_documents(
            [
                SourceDocument(
                    document_id="squad-example",
                    title="Example",
                    context="First sentence. Second sentence.",
                    source_split="train",
                )
            ],
            max_words=10,
            overlap_words=0,
        )
    )

    ensure_collection(client, collection_name, vector_size=2)
    count = index_documents(client, collection_name, FakeEmbedder(), documents, batch_size=10)

    assert count == 1
    assert client.count(collection_name=collection_name, exact=True).count == 1
