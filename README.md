# MCP-Backed RAG Agent

This repository builds one reusable MCP retrieval server over a Qdrant-backed
SQuAD corpus. The first milestone provides a complete SQuAD v1.1 ingestion
path: download, de-duplicate contexts, chunk, embed, and idempotently index.

## Dataset decision

The corpus is [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad).
It contains answerable Wikipedia-based questions with answer spans and a
validation split. Index `train` initially and reserve `validation` for later
retrieval and answer evaluation.

## Indexing design

SQuAD repeats each context once per question. Ingestion de-duplicates those
rows with a stable title/context hash. Qdrant points also have deterministic
IDs, so rerunning ingestion updates rather than duplicates them.

Chunks are sentence-aware, capped at 220 words, with a 40-word overlap. This
keeps retrieved claims readable and preserves citations while handling answers
that straddle a boundary. A sentence longer than the cap is split on word
boundaries. Each Qdrant payload has its title, split, document ID, chunk index,
source offsets, and text.

Embeddings use `BAAI/bge-small-en-v1.5` (384 dimensions) through
`sentence-transformers`. It is a capable small English retrieval baseline that
runs locally and avoids per-request API costs. The tradeoff is the initial
model download and a local CPU/RAM indexing pass; an API model might improve
quality but would add cost and provider dependency.

## Run ingestion

Create and activate an environment, then install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

First verify SQuAD loading and chunking without the model or Qdrant:

```powershell
python -m rag_ingestion.index --max-contexts 10 --dry-run
pytest
```

## MCP server

The indexed corpus is exposed through a single FastMCP server. It has two
tools: `search_documents(query, top_k)` for semantic retrieval and
`filter_by_metadata(title, source_split, limit)` for exact SQuAD metadata
lookups. Both return the chunk text, article title, split, score (for search),
and source offsets.

With the Qdrant environment variables above set, start the Streamable HTTP
server locally:

```powershell
python -m rag_ingestion.server
```

It listens on `http://127.0.0.1:8000/mcp` by default. Set `MCP_HOST` and
`MCP_PORT` for a different bind address or port. The server defers Qdrant and
embedding-model initialization until the first tool call, which keeps startup
and deployment health checks independent of a database request.

The local server has no client authentication. Keep it bound to loopback during
development; deployment-appropriate access control is required before
exposing it on the internet.

## Deploy to Prefect Horizon

Prefect Horizon is the selected host because it is maintained by the
FastMCP team and provides managed HTTPS endpoints, authentication, and
GitHub-driven redeployments. Before deploying, push the latest server changes
to the repository's default branch. In Horizon, create a server from that
repository using this entrypoint:

```text
src/rag_ingestion/server.py:mcp
```

Horizon detects this repository's `pyproject.toml` and installs its declared
dependencies. Add these two runtime secrets in Horizon's deployment settings;
do not commit them to the repository:

```text
QDRANT_URL
QDRANT_API_KEY
```

Enable Horizon authentication before deploying. Horizon will provide an HTTPS
Streamable HTTP endpoint in this form:

```text
https://your-server-name.fastmcp.app/mcp
```

Validate the entrypoint locally before deploy with:

```powershell
.\.venv\Scripts\fastmcp.exe inspect src\rag_ingestion\server.py:mcp
```

Create a Qdrant Cloud free cluster and export the values in `.env.example`,
then run a small connectivity and embedding smoke test:

```powershell
$env:QDRANT_URL = "https://your-cluster.cloud.qdrant.io"
$env:QDRANT_API_KEY = "your-api-key"
python -m rag_ingestion.index --max-contexts 100
```

For the complete training corpus, omit `--max-contexts`:

```powershell
python -m rag_ingestion.index --split train --batch-size 64
```

The collection defaults to `squad_v1_bge_small`; set `QDRANT_COLLECTION` to a
new name when changing embedding models or dimensions. The code refuses an
incompatible existing collection rather than overwriting it. Qdrant free
clusters can suspend after inactivity, so wake the cluster before a demo.

## Verification

```powershell
ruff check .
pytest
```

## LangGraph retrieval agent

The client-side LangGraph workflow connects to the deployed
MCP server, decomposes a question into focused retrieval queries, retrieves
context for each query, and synthesizes an answer grounded in that context.
It does not start or redeploy the MCP server.

Set these client-side variables (preferably in `.env`, which is ignored by
Git):

```text
ANTHROPIC_API_KEY=your-anthropic-api-key
MCP_SERVER_URL=https://qdrant.fastmcp.app/mcp
MCP_AUTH_MODE=oauth
MCP_OAUTH_STORAGE_KEY=choose-a-long-random-local-secret
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agent-mcp
```

The OAuth client requests a confidential dynamic registration so Horizon
returns a client secret for the browser flow. If Horizon expects HTTP Basic
authentication at the token endpoint, set
`MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD=client_secret_basic`; the default is
`client_secret_post`.

OAuth tokens are stored in an encrypted local file store. Set
`MCP_OAUTH_STORAGE_KEY` to a stable, private random value; changing it later
will require signing in again. The default storage directory is `.oauth-v3` in
the project; set `MCP_OAUTH_STORAGE_DIR` to override it. Windows Credential
Manager is not used because Horizon tokens can exceed its credential-blob
limit. Horizon currently advertises `client_secret_basic` for its dynamically
registered client while requiring the secret in the token form body; the
client applies this compatibility adjustment automatically.

The OAuth callback binds to `127.0.0.1` by default. Set
`MCP_OAUTH_CALLBACK_HOST` only if your local browser environment requires a
different loopback hostname.

For a Horizon server protected by OAuth, set `MCP_AUTH_MODE=oauth`. The first
run opens a browser for Horizon sign-in and stores the resulting credentials
locally. For a host that gives you a programmatic Bearer
token instead, use `MCP_AUTH_MODE=bearer` and set `MCP_AUTH_TOKEN`. These are
client-side credentials only; they are never stored or used by the deployed
server.

If Horizon's **Connect** panel provides a pre-registered OAuth application,
also set these client-side variables. Do not replace these with your Qdrant or
Anthropic API keys.

```text
MCP_OAUTH_CLIENT_ID=...
MCP_OAUTH_CLIENT_SECRET=...
```

```powershell
python -m rag_ingestion.agent "How does sleep affect declarative memory?"
```

To verify the OAuth connection before making an Anthropic request, run:

```powershell
python -m rag_ingestion.oauth_probe
```

### LangSmith tracing

LangGraph and LangChain automatically emit traces when the three LangSmith
variables above are set. Run the agent normally:

```powershell
python -m rag_ingestion.agent "How does sleep affect declarative memory?"
```

Then open the `agent-mcp` project in LangSmith. The trace should include
the decomposition step, each MCP retrieval call, and the synthesis model call.
Keep `LANGSMITH_API_KEY` in `.env` only; never commit it. Tracing is disabled
when `LANGSMITH_TRACING` is unset or set to `false`.

## Use with Claude Desktop

After adding the Horizon server from its **Connect** page, open a new Claude
conversation and ask a question normally. Claude will choose the retrieval
tool automatically:

```text
What does the SQuAD corpus say about how sleep affects declarative memory?
```

For a semantic question, Claude uses `search_documents`. You can also ask it
to look up exact metadata, for example:

```text
Show me the first five chunks from the SQuAD article titled "Memory" in the
train split.
```

That uses `filter_by_metadata`, which matches the article title and split
exactly. To inspect the available evidence, ask Claude to include the article
title, source split, and retrieved text in its answer.
