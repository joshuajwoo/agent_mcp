"""Run Horizon's browser-based OAuth flow and list the remote MCP tools."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

from rag_ingestion.agent import (
    DEFAULT_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD,
    OAUTH_CALLBACK_HOST,
    OAUTH_STORAGE_DIR,
    OAUTH_STORAGE_SALT,
    _force_horizon_token_auth_method,
)

load_dotenv()


async def main() -> None:
    server_url = os.environ.get("MCP_SERVER_URL", "").strip()
    if not server_url:
        raise ValueError("Missing required configuration: MCP_SERVER_URL")

    auth = OAuth(
        mcp_url=server_url,
        token_storage=FernetEncryptionWrapper(
            key_value=DiskStore(directory=OAUTH_STORAGE_DIR),
            source_material=os.environ["MCP_OAUTH_STORAGE_KEY"],
            salt=OAUTH_STORAGE_SALT,
        ),
        client_id=os.environ.get("MCP_OAUTH_CLIENT_ID") or None,
        client_secret=os.environ.get("MCP_OAUTH_CLIENT_SECRET") or None,
        callback_host=OAUTH_CALLBACK_HOST,
        additional_client_metadata={
            "token_endpoint_auth_method": os.environ.get(
                "MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD",
                DEFAULT_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD,
            ).lower(),
        },
    )
    _force_horizon_token_auth_method(
        auth,
        os.environ.get(
            "MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD",
            DEFAULT_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD,
        ).lower(),
    )
    async with Client(server_url, auth=auth) as client:
        tools = await client.list_tools()

    print("OAuth succeeded. Remote MCP tools:")
    for tool in tools:
        print(f"- {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())
