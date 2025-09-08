# test run the tools


import os
import asyncio
import uuid
from fastmcp import Client
import sys

# adding MLE_Agent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server import mcp_app


def _render_result(res) -> str:
    """Extract joined text from a CallToolResult-like object for readable prints."""
    try:
        blocks = getattr(res, "content", [])
        texts = [getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"]
        s = "\n".join([t for t in texts if t])
        return s or str(res)
    except Exception:
        return str(res)

async def test_bash_tool_local():
    """Basic sanity test for local Bash tool."""
    # Ensure we don't attempt to use Modal in server
    os.environ["USE_MODAL_SANDBOX"] = "0"

    async with Client(mcp_app) as client:
        res = await client.call_tool("bash", {"command": "echo hello-bash"})
        print("local echo =>\n", _render_result(res))


async def test_edit_tool_local():
    """Create, view, and replace content in a local file using the Edit tool."""
    os.environ["USE_MODAL_SANDBOX"] = "0"

    # Use a file in the repo sandbox folder
    os.makedirs("sandbox", exist_ok=True)
    test_path = os.path.abspath(os.path.join("sandbox", "tool_edit_test.txt"))

    # Clean up any existing file
    try:
        os.remove(test_path)
    except FileNotFoundError:
        pass

    async with Client(mcp_app) as client:
        # Create file
        res1 = await client.call_tool(
            "str_replace_editor",
            {"command": "create", "path": test_path, "file_text": "alpha"},
        )
        print("local edit create =>\n", _render_result(res1))

        # View file
        res2 = await client.call_tool(
            "str_replace_editor", {"command": "view", "path": test_path}
        )
        print("local edit view =>\n", _render_result(res2))

        # Replace content
        res3 = await client.call_tool(
            "str_replace_editor",
            {"command": "str_replace", "path": test_path, "old_str": "alpha", "new_str": "beta"},
        )
        print("local edit replace =>\n", _render_result(res3))


async def test_bash_tool_modal():
    """Basic sanity test for Bash tool running inside a Modal Sandbox."""
    os.environ["USE_MODAL_SANDBOX"] = "1"

    async with Client(mcp_app) as client:
        # 1) Normal stdout
        res = await client.call_tool("bash", {"command": "echo hello-modal"})
        print("modal echo =>\n", _render_result(res))

        # 2) No stdout/stderr (demonstrates '(no output)' fallback)
        res2 = await client.call_tool("bash", {"command": "true"})
        print("modal true (no output) =>\n", _render_result(res2))

        # 3) Stderr-only command (glob miss). In Modal, stderr isn't surfaced by the tool,
        # so this will also show '(no output)' which explains the confusing logs seen earlier.
        res3 = await client.call_tool("bash", {"command": "ls *.py"})
        print("modal ls *.py (stderr only) =>\n", _render_result(res3))

        # 4) Directory listing (should show entries if any)
        res4 = await client.call_tool("bash", {"command": "ls -la"})
        print("modal ls -la =>\n", _render_result(res4))


async def test_edit_tool_modal():
    """Create, view, and replace content in a file inside Modal Sandbox volume '/root/sandbox' (Volume: mle-sandbox)."""
    os.environ["USE_MODAL_SANDBOX"] = "1"

    # Use a unique file in the sandbox-mounted volume to avoid collisions across runs
    test_path = f"/root/sandbox/tool_edit_test_{uuid.uuid4().hex}.txt" 
    async with Client(mcp_app) as client:
        # Create file
        res1 = await client.call_tool(
            "str_replace_editor",
            {"command": "create", "path": test_path, "file_text": "alpha"},
        )
        print('modal edit create =>\n', _render_result(res1))

        # View file
        res2 = await client.call_tool(
            "str_replace_editor", {"command": "view", "path": test_path}
        )
        print('modal edit view =>\n', _render_result(res2))

        # Replace content
        res3 = await client.call_tool(
            "str_replace_editor",
            {"command": "str_replace", "path": test_path, "old_str": "alpha", "new_str": "beta"},
        )
        print('modal edit replace =>\n', _render_result(res3))


        # View file
        res4 = await client.call_tool(
            "str_replace_editor", {"command": "view", "path": '.'}
        )
        print('modal edit view cwd =>\n', _render_result(res4))


async def test_bash_modal_output_variants():
    """
    Demonstrate why 'no stdout output' occurs in Modal:
    - Fresh run directory can be empty, so many commands yield no output.
    - The Modal bash session currently doesn't surface stderr in results.
    """
    os.environ["USE_MODAL_SANDBOX"] = "1"
    async with Client(mcp_app) as client:
        cases = [
            ("pwd", "stdout present"),
            ("true", "no stdout/stderr"),
            ("ls *.py", "stderr only (glob miss)"),
            ("find . -maxdepth 1 -type f", "often empty in fresh run dir"),
        ]
        for cmd, note in cases:
            res = await client.call_tool("bash", {"command": cmd})
            print(f"modal case [{note}] $ {cmd} =>\n", _render_result(res))

async def main():
    # Local tests
    # await test_bash_tool_local()
    # await test_edit_tool_local()
    # Modal-backed tests (requires Modal auth and access)
    #await test_bash_tool_modal()
    await test_bash_modal_output_variants()
    #await test_edit_tool_modal()


if __name__ == "__main__":
    asyncio.run(main())
