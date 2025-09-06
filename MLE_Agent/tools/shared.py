# tools for reading, writing, applying code edits..
# This file contains MCP implementations that copy code from anthropic_tools with proper attribution
# Using only MCP types and standard Python types, no external tool framework dependencies
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionFunctionToolParam
import asyncio
import logging
from mcp.types import Tool

logger = logging.getLogger(__name__)

# Utility functions copied from anthropic_tools/run.py with attribution
TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """Run a shell command asynchronously with a timeout."""
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except asyncio.TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc
        
        
class MCPTool(Tool):
    """
    MCP implementation copying Tool from openai.types.chat.chat_completion_tool_union_param.
    Allows using MCP tools with OpenAI API.
    """
    
    def to_tool_param(self) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.inputSchema
            )
        )

