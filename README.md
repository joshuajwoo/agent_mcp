# MCP-Backed RAG Agent

This project builds a reusable MCP retrieval server over explicitly attributed
philosophy, ethics, and theology corpora stored in Qdrant. It retrieves
sources; it does not present one tradition or annotation scheme as neutral authority.

## Corpus profiles

| Profile | Dataset | Purpose | Important limit |
| --- | --- | --- | --- |
| `philosophy` | [`LisaMegaWatts/philosophy-corpus`](https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus) | Historical philosophy and humanities text | Historical primary texts and reference material are not a philosophical consensus. |
| `ethics` | [`hendrycks/ethics`, `commonsense`](https://huggingface.co/datasets/hendrycks/ethics) | Labeled moral scenarios | An evaluation benchmark, not a source of moral truth. |
| `theology` | [`OpenChristianDataOrg/open-christian-data`, `structured_text`](https://huggingface.co/datasets/OpenChristianDataOrg/open-christian-data) | Historical Christian works with provenance | Christian and historical, not an interfaith or universal-theology collection. |

The `all` profile ingests all three. Each point retains `dataset`, `domain`,
`collection_kind`, `perspective`, `source_hub`, and `source_license`. Theology
records also preserve author, tradition, era, and source ID when provided.
This lets retrieval clients show attribution and filter instead of silently
mixing viewpoints.

Embeddings use `BAAI/bge-small-en-v1.5` (384 dimensions) locally through
`sentence-transformers`.

## Run ingestion

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Start with separate small smoke tests. `--max-contexts` limits each profile;
`all` with a limit of 10 prepares up to 30 source records.

```powershell
python -m rag_ingestion.index --dataset philosophy --max-contexts 10 --dry-run
python -m rag_ingestion.index --dataset ethics --max-contexts 10 --dry-run
python -m rag_ingestion.index --dataset theology --max-contexts 10 --dry-run
python -m rag_ingestion.index --dataset all --max-contexts 10 --dry-run
pytest
```

To index, set `QDRANT_URL` and `QDRANT_API_KEY` and use a fresh collection:

```powershell
$env:QDRANT_COLLECTION = "philosophy_ethics_theology_bge_small"
python -m rag_ingestion.index --dataset all --max-contexts 100
```

Do not run unbounded `all` until you estimate storage and embedding time: the
philosophy corpus is especially large. Index profiles separately when their
scaling and retention policies should differ.

## MCP server

```powershell
python -m rag_ingestion.server
```

It listens on `http://127.0.0.1:8000/mcp` by default. Set `MCP_HOST` and
`MCP_PORT` to change the bind address or port.

`search_documents(query, top_k, dataset, domain)` performs semantic retrieval
with optional corpus/domain filters. `filter_by_metadata` supports exact title,
split, dataset, domain, and tradition filters. Results include their dataset,
domain, and available tradition for accurate citation.

Keep the server bound to loopback during development; add suitable
authentication before exposing it publicly.

## Retrieval agent

Set client variables (preferably in an ignored `.env`):

```text
ANTHROPIC_API_KEY=your-anthropic-api-key
MCP_SERVER_URL=https://your-server.example/mcp
MCP_AUTH_MODE=oauth
MCP_OAUTH_STORAGE_KEY=choose-a-long-random-local-secret
```

```powershell
python -m rag_ingestion.agent "How do Aristotle and Mill differ on virtue?"
```

The agent cites each claim with title and dataset, and does not present ethical
or theological positions as universal consensus.

## Verification

```powershell
ruff check .
pytest
```
