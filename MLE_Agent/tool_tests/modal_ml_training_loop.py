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
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession


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
    env = os.environ.copy()
    env["USE_MODAL_SANDBOX"] = "1"

    params = StdioServerParameters(command="python", args=["mcp_server.py"], env=env)

    # Target path inside the Modal sandbox volume mounted at /root/sandbox
    target_path = "/root/sandbox/train_tiny.py"

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 0) Ensure no stale file remains from a previous run
            _ = await session.call_tool(
                "bash",
                {"command": f"rm -f {target_path}"},
            )

            # 1) Create the training script file in the sandbox
            create_res = await session.call_tool(
                "str_replace_editor",
                {"command": "create", "path": target_path, "file_text": TRAIN_SCRIPT},
            )
            print("Create result:", create_res)

            # 2) Optional: view the file (sanity check)
            view_res = await session.call_tool(
                "str_replace_editor",
                {"command": "view", "path": target_path},
            )
            print("View result:", view_res)

            # 3) Start training in the background (unbuffered python) and peek logs every 5 seconds
            start_res = await session.call_tool(
                "bash",
                {"command": "python -u train_tiny.py", "background": True, "name": "tiny-train"},
            )
            print("Start result:", start_res)

            # Peek and poll a few times
            for i in range(5):
                await asyncio.sleep(5)
                peek_res = await session.call_tool(
                    "bash",
                    {"peek": True, "name": "tiny-train", "lines": 50},
                )
                print(f"Peek {i+1}:", peek_res)
                poll_res = await session.call_tool(
                    "bash",
                    {"poll": True, "name": "tiny-train"},
                )
                print(f"Poll {i+1}:", poll_res)

            # Optionally stop the job (training script will exit on its own soon)
            stop_res = await session.call_tool(
                "bash",
                {"stop": True, "name": "tiny-train"},
            )
            print("Stop result:", stop_res)


if __name__ == "__main__":
    asyncio.run(main())
