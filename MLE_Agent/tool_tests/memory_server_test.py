import os
import asyncio
from fastmcp import Client

from MLE_Agent.tools.format_tool import format_print_tool


async def main():
    # Use a single FastMCP client connected to both servers
    config = {
        "mcpServers": {
            "main": {"command": "python", "args": ["mcp_server.py"], "env": {"USE_MODAL_SANDBOX": "1"}},
            "memory": {"command": "python", "args": ["memory_server.py"]},
        }
    }

    async with Client(config) as client:
        # First touch the main server bash to ensure run dir exists
        args = {}
        start_res = await client.call_tool("main_bash", args)
        format_print_tool("main_bash", args, start_res)

        # Now exercise memory tools (memory server lazily resolves run dir to latest)
        # Create entities
        args = {
            "entities": [
                {"name": "Alice", "entityType": "Person", "observations": ["Likes coffee"]},
                {"name": "Acme", "entityType": "Company", "observations": ["Makes widgets"]},
            ]
        }
        res = await client.call_tool("memory_create_entities", args)
        format_print_tool("memory_create_entities", args, res)

        # Create relation
        args = {
            "relations": [
                {"from": "Alice", "to": "Acme", "relationType": "works_at"}
            ]
        }
        res = await client.call_tool("memory_create_relations", args)
        format_print_tool("memory_create_relations", args, res)

        # Add observations
        args = {
            "observations": [
                {"entityName": "Alice", "contents": ["Enjoys hiking", "Likes coffee"]}
            ]
        }
        res = await client.call_tool("memory_add_observations", args)
        format_print_tool("memory_add_observations", args, res)

        # Read graph
        args = {}
        res = await client.call_tool("memory_read_graph", args)
        format_print_tool("memory_read_graph", args, res)

        # Search nodes
        args = {"query": "alice"}
        res = await client.call_tool("memory_search_nodes", args)
        format_print_tool("memory_search_nodes", args, res)

        # Open nodes
        args = {"names": ["Alice"]}
        res = await client.call_tool("memory_open_nodes", args)
        format_print_tool("memory_open_nodes", args, res)

        # Delete observation
        args = {"deletions": [{"entityName": "Alice", "observations": ["Likes coffee"]}]}
        res = await client.call_tool("memory_delete_observations", args)
        format_print_tool("memory_delete_observations", args, res)

        # Delete relation
        args = {"relations": [{"from": "Alice", "to": "Acme", "relationType": "works_at"}]}
        res = await client.call_tool("memory_delete_relations", args)
        format_print_tool("memory_delete_relations", args, res)

        # Delete entity
        args = {"entityNames": ["Alice"]}
        res = await client.call_tool("memory_delete_entities", args)
        format_print_tool("memory_delete_entities", args, res)

        # Final read
        args = {}
        res = await client.call_tool("memory_read_graph", args)
        format_print_tool("memory_read_graph", args, res)


if __name__ == "__main__":
    asyncio.run(main())
