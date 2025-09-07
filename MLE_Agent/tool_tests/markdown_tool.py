import os
import asyncio
from fastmcp import Client
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(sys.path)

from tools.format_tool import format_print_tool
from mcp_server import mcp_app

async def main():
    os.environ["USE_MODAL_SANDBOX"] = "1"
    async with Client(mcp_app) as client:
        args = dict(url="https://arxiv.org/pdf/2505.14669", page_range="1-2")
        output = await client.call_tool("pdf_to_markdown", args)
        format_print_tool("pdf_to_markdown", args, output)
            

if __name__ == "__main__":
    asyncio.run(main())
