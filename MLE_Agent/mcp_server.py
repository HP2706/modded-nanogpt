# MCP server implementation using the MCP tools
# This file sets up the MCP server with our custom tools

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from mcp.types import TextContent, ImageContent, EmbeddedResource
import mcp.types as types
from mcp.server import Server
from pydantic import AnyUrl

# Import our MCP tools
from tools.edit import MCPEditTool
from tools.bash import MCPBashTool
import os
import modal

logger = logging.getLogger(__name__)

app = Server("modal-server")

# Initialize MCP tools
# Optionally run tools inside a shared Modal Sandbox when env is set
_use_modal = os.environ.get("USE_MODAL_SANDBOX", "0") == "1"
_shared_sandbox = None
if _use_modal:
        
    modal_app = modal.App.lookup("mle-agent-tools", create_if_missing=True)

    image = modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", 
        add_python="3.12"
    ).uv_pip_install(
        'torch',
        'transformers',
        'datasets',
        'einops',
        'psutil',
        'hf-transfer',
        'fire',
        'einx',
        'matplotlib',
    )


    try:
        _shared_sandbox = modal.Sandbox.create(
            app=modal_app,
            image=image,
            volumes={
                "/root/sandbox": modal.Volume.from_name("mle-sandbox", create_if_missing=True)
            },
            timeout=30
        )
    except Exception as e:
        logger.warning(f"Could not start Modal Sandbox, falling back to local tools: {e}")
        _shared_sandbox = None

bash_tool = MCPBashTool(sandbox=_shared_sandbox) if _shared_sandbox else MCPBashTool()
edit_tool = MCPEditTool(sandbox=_shared_sandbox) if _shared_sandbox else MCPEditTool()
# List of available tools for direct access
TOOLS = [bash_tool, edit_tool]


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """List available resources."""
    return []


@app.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    try:
        return json.dumps({"result": "example"}, indent=2)
    except Exception as e:
        raise RuntimeError(f"API error: {str(e)}")


@app.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return []


@app.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    raise ValueError(f"Unknown prompt: {name}")


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return TOOLS


@app.call_tool()
async def call_tool(
    name: str, arguments: Any
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls for MCP tools."""
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a dictionary")

    try:
        if name == "bash":
            print('executing bash tool with arguments:', arguments)
            result = await bash_tool.execute(arguments)
        elif name == "str_replace_editor":
            print('executing str_replace_editor tool with arguments:', arguments)
            result = await edit_tool.execute(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        # Return the result directly since tools now return TextContent sequences
        return result

    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}")
        return [
            TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")
        ]

async def main():
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
