import asyncio

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_ingestion.dataset import SourceDocument, build_index_documents
from rag_ingestion.index import ensure_collection, index_documents
from rag_ingestion.retrieval import QdrantRetriever
from rag_ingestion.server import create_server

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


class FakeEmbedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.array([[float(len(text)), 1.0] for text in texts])


def _retriever() -> QdrantRetriever:
    client = QdrantClient(":memory:")
    collection_name = "retrieval_test"
    documents = list(
        build_index_documents(
            [
                SourceDocument(
                    document_id="squad-example",
                    title="Example",
                    context="First fact. Second fact.",
                    source_split="train",
                )
            ],
            max_words=10,
            overlap_words=0,
        )
    )
    ensure_collection(client, collection_name, vector_size=2)
    index_documents(client, collection_name, FakeEmbedder(), documents, batch_size=10)
    return QdrantRetriever(client, collection_name, FakeEmbedder())


def test_retriever_searches_and_filters_qdrant_payloads() -> None:
    retriever = _retriever()

    search_results = retriever.search("What is the first fact?", top_k=1)
    filtered_results = retriever.filter_by_metadata(title="Example", source_split=None, limit=10)

    assert search_results[0].title == "Example"
    assert search_results[0].score >= 0
    assert filtered_results[0].source_split == "train"


def test_mcp_server_exposes_the_retrieval_tools() -> None:
    server = create_server(lambda: _retriever())

    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {"search_documents", "filter_by_metadata"}
