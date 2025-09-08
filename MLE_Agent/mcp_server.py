"""
FastMCP-based MCP server exposing bash, edit, and pdf conversion tools.
Preserves Modal Sandbox support (USE_MODAL_SANDBOX=1) and tool names.
"""

import os
import logging
from typing import Any

from fastmcp import FastMCP

from tools.bash import BashContainer
from tools.edit import EditContainer
from tools.pdf import PdfContainer
from tools.shared import agent_volume, fineweb10B_volume
import modal

logger = logging.getLogger(__name__)

mcp_app = FastMCP("modal-server")

# Optional Modal Sandbox for bash/edit tools
_use_modal = os.environ.get("USE_MODAL_SANDBOX", "1") == "1"  # default to true
_shared_sandbox = None

if _use_modal:
    
    with agent_volume.batch_upload(force=True) as batch:
        assert os.path.exists("modded-nanogpt.py"), "modded-nanogpt.py not found"
        batch.put_file("modded-nanogpt.py", "modded-nanogpt.py") 
    
    try:
        modal_app = modal.App.lookup("mle-agent-tools", create_if_missing=True)
        image = modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
        ).uv_pip_install(
            "torch",
            "transformers",
            "datasets",
            "einops",
            "psutil",
            "hf-transfer",
            "fire",
            "einx",
            "matplotlib",
        )
        _shared_sandbox = modal.Sandbox.create(
            app=modal_app,
            image=image,
            volumes={
                "/root/sandbox": agent_volume,
                "/root/fineweb10B": fineweb10B_volume,
            },
            timeout=120,
        )
    except Exception as e:
        logger.warning(
            f"Could not start Modal Sandbox, falling back to local tools: {e}"
        )
        _shared_sandbox = None

_bash_state = BashContainer(sandbox=_shared_sandbox) if _shared_sandbox else BashContainer()
_edit_state = EditContainer(sandbox=_shared_sandbox) if _shared_sandbox else EditContainer()
_pdf_state = PdfContainer(bash_container=_bash_state, edit_container=_edit_state)

# Register object methods directly so they can share state
# bash tools
mcp_app.tool(_bash_state.run_command)
mcp_app.tool(_bash_state.run_command_background)
mcp_app.tool(_bash_state.stop_background_job)
mcp_app.tool(_bash_state.poll_background_job)
mcp_app.tool(_bash_state.list_background_jobs)
mcp_app.tool(_bash_state.restart_session)

# edit tools
mcp_app.tool(_edit_state.str_replace)
mcp_app.tool(_edit_state.view)
mcp_app.tool(_edit_state.insert)
mcp_app.tool(_edit_state.read_file)
mcp_app.tool(_edit_state.write_file)

mcp_app.tool(_pdf_state.pdf_to_markdown)

if __name__ == "__main__":
    # Default to stdio transport; FastMCP CLI can also run this script directly
    mcp_app.run()
