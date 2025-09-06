# test run the tools


import os
import asyncio
import uuid
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession


async def test_bash_tool_local():
    """Basic sanity test for local Bash tool."""
    # Ensure we don't attempt to use Modal in server
    env = os.environ.copy()
    env["USE_MODAL_SANDBOX"] = "0"

    params = StdioServerParameters(command="python", args=["mcp_server.py"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("bash", {"command": "echo hello-bash"})
            print(res)


async def test_edit_tool_local():
    """Create, view, and replace content in a local file using the Edit tool."""
    env = os.environ.copy()
    env["USE_MODAL_SANDBOX"] = "0"

    # Use a file in the repo sandbox folder
    os.makedirs("sandbox", exist_ok=True)
    test_path = os.path.abspath(os.path.join("sandbox", "tool_edit_test.txt"))

    # Clean up any existing file
    try:
        os.remove(test_path)
    except FileNotFoundError:
        pass

    params = StdioServerParameters(command="python", args=["mcp_server.py"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create file
            res1 = await session.call_tool(
                "str_replace_editor",
                {"command": "create", "path": test_path, "file_text": "alpha"},
            )
            print(res1)

            # View file
            res2 = await session.call_tool(
                "str_replace_editor", {"command": "view", "path": test_path}
            )
            print(res2)

            # Replace content
            res3 = await session.call_tool(
                "str_replace_editor",
                {"command": "str_replace", "path": test_path, "old_str": "alpha", "new_str": "beta"},
            )
            print(res3)


async def test_bash_tool_modal():
    """Basic sanity test for Bash tool running inside a Modal Sandbox."""
    env = os.environ.copy()
    env["USE_MODAL_SANDBOX"] = "1"

    params = StdioServerParameters(command="python", args=["mcp_server.py"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("bash", {"command": "echo hello-modal"})
            print(res)


async def test_edit_tool_modal():
    """Create, view, and replace content in a file inside Modal Sandbox volume '/root/sandbox' (Volume: mle-sandbox)."""
    env = os.environ.copy()
    env["USE_MODAL_SANDBOX"] = "1"

    # Use a unique file in the sandbox-mounted volume to avoid collisions across runs
    test_path = f"/root/sandbox/tool_edit_test_{uuid.uuid4().hex}.txt"

    params = StdioServerParameters(command="python", args=["mcp_server.py"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create file
            res1 = await session.call_tool(
                "str_replace_editor",
                {"command": "create", "path": test_path, "file_text": "alpha"},
            )
            print(res1)

            # View file
            res2 = await session.call_tool(
                "str_replace_editor", {"command": "view", "path": test_path}
            )
            print(res2)

            # Replace content
            res3 = await session.call_tool(
                "str_replace_editor",
                {"command": "str_replace", "path": test_path, "old_str": "alpha", "new_str": "beta"},
            )
            print(res3)


async def main():
    # Local tests
    #await test_bash_tool_local()
    #await test_edit_tool_local()
    # Modal-backed tests (requires Modal auth and access)
    await test_bash_tool_modal()
    await test_edit_tool_modal()


if __name__ == "__main__":
    asyncio.run(main())
