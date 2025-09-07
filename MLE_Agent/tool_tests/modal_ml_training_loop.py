"""
Sequence of MCP tool calls to:
1) Create a Python file in the Modal sandbox volume
2) Write a tiny CPU-only PyTorch training script
3) Execute the training via the Bash tool (stateful Modal session)

Requires:
- Modal auth configured in environment
- The MCP server in this repo (mcp_server.py)
- USE_MODAL_SANDBOX=1 so tools run inside the Modal Sandbox with Volume "mle-sandbox" mounted at /root/sandbox
"""

import os
import asyncio
from fastmcp import Client
import sys
sys.path.append('..')

from MLE_Agent.tools.format_tool import format_print_tool


TRAIN_SCRIPT = r"""
import torch
from torch import nn, optim


def main():
    torch.manual_seed(0)
    # Tiny synthetic regression dataset (CPU-only)
    X = torch.randn(256, 10)
    y = torch.randn(256, 1)

    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    loss_fn = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(5):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
"""


async def main():
    os.environ["USE_MODAL_SANDBOX"] = "1"

    # Will resolve inside the bash session's run directory
    target_path = None

    async with Client("mcp_server.py") as client:
        # 0) Discover the bash session run_dir via startup message and ensure no stale file remains
        args = {}
        start_res = await client.call_tool("bash", args)
        format_print_tool("bash", args, start_res)
        # Expect: "tool started. cwd: <run_dir>" or "tool has been restarted. cwd: <run_dir>"
        msg = start_res.content[0].text
        run_dir = "/root/sandbox"
        if "cwd:" in msg:
            try:
                run_dir = msg.split("cwd:", 1)[1].strip()
            except Exception:
                pass
        target_path = f"{run_dir}/train_tiny.py"
        args = {"command": f"rm -f {target_path}"}
        _ = await client.call_tool("bash", args)

        # 1) Create the training script file in the sandbox
        args = {"command": "create", "path": target_path, "file_text": TRAIN_SCRIPT}
        create_res = await client.call_tool(
            "str_replace_editor",
            args,
        )
        format_print_tool("str_replace_editor", args, create_res)

        # 2) Optional: view the file (sanity check)
        args = {"command": "view", "path": target_path}
        view_res = await client.call_tool(
            "str_replace_editor",
            args,
        )
        format_print_tool("str_replace_editor", args, view_res)

        # 3) Start training in the background (unbuffered python) and peek logs every 5 seconds
        args = {"command": f"python -u {target_path}", "background": True, "name": "tiny-train"}
        start_res = await client.call_tool(
            "bash",
            args,
        )
        format_print_tool("bash", args, start_res)

        # Peek and poll a few times
        for i in range(5):
            await asyncio.sleep(5)
            args_peek = {"peek": True, "name": "tiny-train", "lines": 50}
            peek_res = await client.call_tool(
                "bash",
                args_peek,
            )
            format_print_tool("bash", args_peek, peek_res)
            args_poll = {"poll": True, "name": "tiny-train"}
            poll_res = await client.call_tool(
                "bash",
                args_poll,
            )
            format_print_tool("bash", args_poll, poll_res)

        # Optionally stop the job (training script will exit on its own soon)
        args = {"stop": True, "name": "tiny-train"}
        stop_res = await client.call_tool(
            "bash",
            args,
        )
        format_print_tool("bash", args, stop_res)


if __name__ == "__main__":
    asyncio.run(main())
