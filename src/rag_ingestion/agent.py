"""LangGraph client that retrieves from the deployed MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class AgentSettings:
    """Configuration for the local LangGraph client."""

    mcp_server_url: str
    mcp_auth_token: str | None
    anthropic_model: str
    top_k: int

    @classmethod
    def from_environment(cls) -> AgentSettings:
        top_k = int(os.environ.get("RETRIEVAL_TOP_K", str(DEFAULT_TOP_K)))
        if not 1 <= top_k <= 20:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20")
        return cls(
            mcp_server_url=os.environ.get("MCP_SERVER_URL", "").strip(),
            mcp_auth_token=os.environ.get("MCP_AUTH_TOKEN") or None,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            top_k=top_k,
        )

    def validate(self) -> None:
        if not self.mcp_server_url:
            raise ValueError("Missing required configuration: MCP_SERVER_URL")
        if not self.mcp_server_url.startswith(("https://", "http://")):
            raise ValueError("MCP_SERVER_URL must be an HTTP(S) URL")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Missing required configuration: ANTHROPIC_API_KEY")


class SubquestionPlan(BaseModel):
    """Structured result from the question-decomposition step."""

    subquestions: list[str] = Field(min_length=1, max_length=4)


class RetrievalState(TypedDict):
    """Mutable graph state shared by the explicit retrieval workflow."""

    question: str
    subquestions: list[str]
    retrievals: list[dict[str, str]]
    answer: str


DECOMPOSITION_PROMPT = """Break the user's question into one to four independent
retrieval queries. Preserve the original wording where it is already a focused
question. Do not answer the question and do not add facts that were not asked.
"""

SYNTHESIS_PROMPT = """Answer the user's question using only the retrieved MCP
tool results below. State when the results do not support an answer. Cite the
SQuAD article title in square brackets for each factual claim, such as
[Memory]. Do not invent sources or use outside knowledge.
"""


def _mcp_connection(settings: AgentSettings) -> dict[str, dict[str, object]]:
    connection: dict[str, object] = {
        "transport": "http",
        "url": settings.mcp_server_url,
    }
    if settings.mcp_auth_token:
        connection["headers"] = {
            "Authorization": f"Bearer {settings.mcp_auth_token}",
        }
    return {"squad_retrieval": connection}


def _tool_result_to_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(getattr(result, "content"))
    return json.dumps(result, default=str)


async def build_retrieval_agent(settings: AgentSettings | None = None) -> object:
    """Build a LangGraph workflow backed by the remote MCP retrieval tools."""
    runtime_settings = settings or AgentSettings.from_environment()
    runtime_settings.validate()

    mcp_client = MultiServerMCPClient(_mcp_connection(runtime_settings))
    tools = await mcp_client.get_tools(server_name="squad_retrieval")
    search_tool = next((tool for tool in tools if tool.name == "search_documents"), None)
    if search_tool is None:
        names = ", ".join(tool.name for tool in tools)
        raise RuntimeError(f"Remote MCP server did not expose search_documents (found: {names})")

    model = ChatAnthropic(model=runtime_settings.anthropic_model, temperature=0)
    planner = create_agent(
        model=model,
        tools=[],
        response_format=SubquestionPlan,
        system_prompt=DECOMPOSITION_PROMPT,
    )

    async def decompose(state: RetrievalState) -> dict[str, object]:
        planner_result = await planner.ainvoke(
            {"messages": [HumanMessage(content=state["question"])]}
        )
        plan = planner_result["structured_response"]
        subquestions = list(plan.subquestions)
        return {"subquestions": subquestions}

    async def retrieve(state: RetrievalState) -> dict[str, object]:
        subquestions: list[str] = state["subquestions"]
        responses = await asyncio.gather(
            *(
                search_tool.ainvoke({"query": question, "top_k": runtime_settings.top_k})
                for question in subquestions
            )
        )
        return {
            "retrievals": [
                {"subquestion": question, "result": _tool_result_to_text(result)}
                for question, result in zip(subquestions, responses, strict=True)
            ]
        }

    async def synthesize(state: RetrievalState) -> dict[str, object]:
        context = json.dumps(state["retrievals"], indent=2)
        response = await model.ainvoke(
            [
                SystemMessage(content=SYNTHESIS_PROMPT),
                HumanMessage(
                    content=f"Question: {state['question']}\n\nRetrieved results:\n{context}"
                ),
            ]
        )
        return {"answer": str(response.content)}

    graph = StateGraph(RetrievalState)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


async def answer_question(question: str, settings: AgentSettings | None = None) -> str:
    """Run the Phase 4 workflow and return its grounded answer."""
    if not question.strip():
        raise ValueError("question must not be empty")
    agent = await build_retrieval_agent(settings)
    result = await agent.ainvoke({"question": question})
    return str(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the remote SQuAD retrieval agent a question.")
    parser.add_argument("question", help="Question to decompose, retrieve, and answer")
    args = parser.parse_args()
    print(asyncio.run(answer_question(args.question)))


if __name__ == "__main__":
    main()
