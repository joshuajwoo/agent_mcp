import asyncio

import pytest

from rag_ingestion.agent import (
    AgentSettings,
    _mcp_connection,
    _tool_result_to_text,
    build_retrieval_agent,
)


def test_agent_settings_read_remote_server_and_optional_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_SERVER_URL", "https://qdrant.fastmcp.app/mcp")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")

    settings = AgentSettings.from_environment()

    assert settings.mcp_server_url == "https://qdrant.fastmcp.app/mcp"
    assert settings.top_k == 3
    assert _mcp_connection(settings) == {
        "squad_retrieval": {
            "transport": "http",
            "url": "https://qdrant.fastmcp.app/mcp",
            "headers": {"Authorization": "Bearer test-token"},
        }
    }


def test_agent_settings_require_url_and_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MCP_SERVER_URL"):
        AgentSettings.from_environment().validate()

    monkeypatch.setenv("MCP_SERVER_URL", "https://qdrant.fastmcp.app/mcp")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AgentSettings.from_environment().validate()


def test_agent_settings_reject_invalid_retrieval_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "21")
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        AgentSettings.from_environment()


def test_tool_result_conversion_preserves_text_content() -> None:
    assert _tool_result_to_text("already text") == "already text"
    assert _tool_result_to_text({"title": "Memory"}) == '{"title": "Memory"}'


def test_build_retrieval_agent_constructs_decompose_retrieve_synthesize_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSearchTool:
        name = "search_documents"

    class FakeMcpClient:
        def __init__(self, _: object) -> None:
            pass

        async def get_tools(self, *, server_name: str) -> list[FakeSearchTool]:
            assert server_name == "squad_retrieval"
            return [FakeSearchTool()]

    monkeypatch.setattr("rag_ingestion.agent.MultiServerMCPClient", FakeMcpClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    graph = asyncio.run(
        build_retrieval_agent(
            AgentSettings(
                mcp_server_url="https://qdrant.fastmcp.app/mcp",
                mcp_auth_token=None,
                anthropic_model="claude-sonnet-4-5",
                top_k=4,
            )
        )
    )

    assert {node.id for node in graph.get_graph().nodes.values()} >= {
        "__start__",
        "decompose",
        "retrieve",
        "synthesize",
        "__end__",
    }
