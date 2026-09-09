# Project Plan: MCP-Backed RAG Agent (LangGraph + Claude, Shared via One MCP Server)

## Overview

A single MCP server wraps a Qdrant-backed retrieval tool. Two independent
clients consume it — a LangGraph agent (via LangChain) and Claude
(Desktop/claude.ai) — proving the tool is genuinely reusable across hosts,
not hardcoded into one app. LangSmith provides tracing; an evaluation
script (TBD) measures retrieval and answer quality.

Kept deliberately separate from this project: an existing AgentDojo-style
clone (LLM agent security / prompt-injection evaluation) — a distinct
resume line, not merged into this one.

## Architecture

```
                     ┌─────────────────────┐
   LangGraph Agent ──┤                     │
   (LangChain v1.0,  │   MCP Server        │
    langchain-mcp-   │   (FastMCP, deployed│──── Qdrant Cloud
    adapters)        │   remotely, Stream- │     (free tier cluster)
                      │   able HTTP)        │
   Claude Desktop /───┤                     │
   claude.ai connector└─────────────────────┘
        (same URL, second independent client)
```

---

## Phase 0 — Pick the dataset

Recommended: a Hugging Face dataset with built-in ground-truth Q&A pairs —
**SQuAD**, **Natural Questions**, or **MS MARCO**. This solves the
evaluation bottleneck for free: pull 30-50 existing questions as the eval
set instead of hand-writing them.

Alternative (more "portfolio-personal", weaker on free eval data): a
Wikipedia subset or arXiv abstracts dataset, framed as a domain-specific
research assistant.

**Decision:** _(fill in once chosen)_

---

## Phase 1 — Data ingestion & indexing

- [ ] Pull dataset via `datasets` library (`load_dataset(...)`)
- [ ] Decide + document a chunking strategy (fixed-size+overlap vs.
      sentence-aware) — be ready to explain the choice, not just apply a
      default
- [ ] Pick an embedding model (open-source via `sentence-transformers`, or
      API-based) and note the tradeoff
- [ ] Batch-upsert into a Qdrant Cloud free cluster (1 GB RAM, 0.5 vCPU,
      4 GB disk, single node, no credit card required)

**Gotcha:** free Qdrant clusters suspend after a week of inactivity —
"wake" the cluster before any demo, or add the keep-alive stretch goal
below.

---

## Phase 2 — MCP server (the reusable tool layer)

Build with FastMCP. Target 2-3 tools:

- [ ] `search_documents(query, top_k)` — core semantic search
- [ ] `filter_by_metadata(...)` — if the dataset has usable metadata
- [ ] (optional) a re-rank step as a third tool

Use **Streamable HTTP transport** (current standard for remote MCP
servers), not stdio — this is what makes the server reachable by both
clients over the internet instead of as a local subprocess.

---

## Phase 3 — Deploy the MCP server remotely

Two reasonable options:

- **Prefect Horizon** — built by the FastMCP team, `git push` → live MCP
  URL. Most stack-aligned choice since the server is already FastMCP.
- **Fly.io** — DIY alternative, good cost optimization via auto-suspend
  for low-traffic portfolio use.

Output of this phase: one HTTPS URL, used by both clients below.

- [ ] Deploy server
- [ ] Confirm URL is reachable and returns a valid `tools/list` response

---

## Phase 4 — Client 1: LangGraph agent (via LangChain)

- [ ] Build agent with `create_agent` (LangChain v1.0, LangGraph runtime
      under the hood)
- [ ] Connect to the deployed MCP server via `langchain-mcp-adapters`
- [ ] Implement **decompose → retrieve-per-subquestion → synthesize** —
      the concrete justification for using an agent/graph instead of a
      single tool call
- [ ] Enable **LangSmith tracing** on the full run

---

## Phase 5 — Client 2: Claude connects to the same server

- [ ] Add the deployed server's URL as a custom connector in Claude
      (Settings → Connectors → Add custom connector)
- [ ] Ask a retrieval-requiring question and confirm it calls the same
      `search_documents` tool
- [ ] Screen-record or screenshot this — this is the actual proof of the
      "one server, two hosts" claim

---

## Phase 6 — Evaluation (framework TBD)

- [ ] Fix a 30-50 question eval set (from Phase 0's dataset if it has
      ground truth)
- [ ] Run naive single-shot RAG vs. the Phase 4 decompose-and-synthesize
      version on the same set — report a before/after, not just one score
- [ ] Document the limitation: these metrics measure internal consistency
      (did the answer match the retrieved context) — not whether the
      retrieved context was factually true

**Framework decision:** _(pending — Ragas / DeepEval / other)_

---

## Phase 7 — Write-up / packaging

- [ ] README with architecture diagram, screenshots of both clients
      hitting the same tool, eval results table
- [ ] "Tradeoffs" section: why Qdrant over Pinecone, why LangGraph over a
      single chain, why MCP was worth it with only two real consumers

---

## Optional stretch goals (Ignore these for now)

- [ ] Swappable Pinecone backend behind the same MCP tool interface (gets
      both vectorDB keywords on the resume off one component)
- [ ] Keep-alive script pinging Qdrant weekly to prevent free-tier
      suspension before a demo
