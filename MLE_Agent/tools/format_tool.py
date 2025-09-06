from mcp.types import TextContent, CallToolResult
from pprint import pprint

def format_print_tool(tool_name: str, tool_args: dict, res: CallToolResult) -> None:
    data = {
        'tool_name': tool_name,
        'tool_args': tool_args,
        **res.model_dump(),
    }
    pprint(data)