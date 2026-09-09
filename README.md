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

Phase 2 exposes the indexed corpus through a single FastMCP server. It has two
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
development; Phase 3 should add deployment-appropriate access control before
exposing it on the internet.

## Deploy to Prefect Horizon

Prefect Horizon is the selected Phase 3 host because it is maintained by the
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

For the complete Phase 1 corpus, omit `--max-contexts`:

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
