"""
FastMCP-based MCP server exposing bash, edit, and pdf conversion tools.
Preserves Modal Sandbox support (USE_MODAL_SANDBOX=1) and tool names.
"""

import os
import logging
import asyncio
from fastmcp import FastMCP
from tools.bash import BashContainer
from tools.edit import EditContainer
from tools.pdf import PdfContainer
from tools.shared import agent_volume, fineweb10B_volume, LazySandBox
from tools.memory import MemoryContainer
import modal
from modal import Secret

logger = logging.getLogger(__name__)

mcp_app = FastMCP("modal-server")

# Optional Modal Sandbox for bash/edit tools
_use_modal = os.environ.get("USE_MODAL_SANDBOX", "1") == "1"  # default to true
_shared_sandbox = None

if _use_modal:
    # if we are not providing a run dir, we need to upload the modded-nanogpt.py file
    with agent_volume.batch_upload(force=True) as batch:
        #batch.put_file("environments/modded_nanogpt.py", "/root/modded_nanogpt.py") 
        batch.put_file("environments/modded_nanogpt_unoptimized.py", "/root/modded_nanogpt_unoptimized.py")
        batch.put_file("environments/cifar_speedrun_unoptimized.py", "/root/cifar_speedrun.py") # RENAME FILE DELIBERATELY
        #batch.put_file('environments/cifar_speedrun.py', '/root/cifar_speedrun.py')
    
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
            "wandb",
        )
        
        kwargs = {
            "app": modal_app,
            "image": image,
            "volumes": {
                "/root/sandbox": agent_volume,
                "/root/fineweb10B": fineweb10B_volume,
            },
            "timeout": 60*5,
            "secrets": [Secret.from_name("wandb")],
            #'gpu': 'A100-80GB:1'
        }
        _shared_sandbox = LazySandBox(**kwargs)
    except Exception as e:
        logger.warning(
            f"Could not start Modal Sandbox, falling back to local tools: {e}"
        )
        _shared_sandbox = None
        

#_run_dir_env = os.environ.get("RUN_DIR") or os.environ.get("DEMO_RUN_DIR")

_run_dir_env = "/root/sandbox/runs/2025-09-08_19-09-54"

_bash_state = (
    BashContainer(sandbox=_shared_sandbox, automount_path="/root/sandbox", run_dir=_run_dir_env)
    if _shared_sandbox
    else BashContainer(run_dir=_run_dir_env)
)

async def get_automount_path():
    return await _bash_state.ensure_cwd()

automount_path = asyncio.run(get_automount_path())

_edit_state = (
    EditContainer(
        sandbox=_shared_sandbox,
        automount_path=automount_path,
    )
    if _shared_sandbox
    else EditContainer(automount_path=automount_path)
)

_pdf_state = PdfContainer(
    bash_container=_bash_state, 
    edit_container=_edit_state,
    shared_sandbox=_shared_sandbox
)

_memory_state = (
    MemoryContainer(
        sandbox=_shared_sandbox,
        automount_path=automount_path,
    )
    if _shared_sandbox
    else MemoryContainer(
        automount_path=automount_path,
    )
)



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

# memory tools
ADD_MEMORY_TOOLS = False #os.environ.get("ADD_MEMORY_TOOLS", "1") == "1"

if ADD_MEMORY_TOOLS:
    mcp_app.tool(_memory_state.read_graph)
    mcp_app.tool(_memory_state.create_entities)
    mcp_app.tool(_memory_state.create_relations)
    mcp_app.tool(_memory_state.add_observations)
    mcp_app.tool(_memory_state.delete_entities)
    mcp_app.tool(_memory_state.delete_observations)
    mcp_app.tool(_memory_state.delete_relations)
    mcp_app.tool(_memory_state.search_nodes)
    mcp_app.tool(_memory_state.open_nodes)

if __name__ == "__main__":
    # Default to stdio transport; FastMCP CLI can also run this script directly
    mcp_app.run()
