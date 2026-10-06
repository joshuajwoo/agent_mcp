"""Streamable HTTP MCP server exposing attributed, multi-corpus retrieval."""

from __future__ import annotations

import os
from collections.abc import Callable

from fastmcp import FastMCP

from rag_ingestion.retrieval import QdrantRetriever


def create_server(
    retriever_factory: Callable[[], QdrantRetriever] = QdrantRetriever.from_environment,
) -> FastMCP:
    """Build the MCP server without opening Qdrant until a tool is invoked."""
    mcp = FastMCP("Philosophy, Ethics & Theology Retrieval")
    retriever: QdrantRetriever | None = None

    def get_retriever() -> QdrantRetriever:
        nonlocal retriever
        if retriever is None:
            retriever = retriever_factory()
        return retriever

    @mcp.tool
    def search_documents(
        query: str, top_k: int = 5, dataset: str | None = None, domain: str | None = None
    ) -> dict[str, object]:
        """Find attributed source chunks; optionally limit results to a corpus or domain."""
        results = get_retriever().search(query, top_k, dataset=dataset, domain=domain)
        return {"query": query, "results": [result.to_dict() for result in results]}

    @mcp.tool
    def filter_by_metadata(
        title: str | None = None,
        source_split: str | None = None,
        dataset: str | None = None,
        domain: str | None = None,
        tradition: str | None = None,
        limit: int = 10,
    ) -> dict[str, object]:
        """List chunks by exact source metadata, including corpus and tradition when present."""
        results = get_retriever().filter_by_metadata(
            title=title,
            source_split=source_split,
            dataset=dataset,
            domain=domain,
            tradition=tradition,
            limit=limit,
        )
        return {
            "title": title,
            "source_split": source_split,
            "dataset": dataset,
            "domain": domain,
            "tradition": tradition,
            "results": [result.to_dict() for result in results],
        }

    return mcp


mcp = create_server()


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host=os.environ.get("MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("MCP_PORT", "8000")),
    )
