# tools for reading, writing, applying code edits..
# This file contains MCP implementations that copy code from anthropic_tools with proper attribution
# Using only MCP types and standard Python types, no external tool framework dependencies
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionFunctionToolParam
import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence, get_args

from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import PrivateAttr

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


# Copied from anthropic_tools/bash.py with attribution
class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise RuntimeError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str) -> Sequence[TextContent]:
        """Execute a command in the bash shell."""
        if not self._started:
            raise RuntimeError("Session has not started.")
        if self._process.returncode is not None:
            return [
                TextContent(type="text", text="tool must be restarted"),
                TextContent(type="text", text=f"Error: bash has exited with returncode {self._process.returncode}")
            ]
        if self._timed_out:
            raise RuntimeError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted"
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise RuntimeError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted"
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        contents = []
        if output:
            contents.append(TextContent(type="text", text=output))
        if error:
            contents.append(TextContent(type="text", text=f"Error: {error}"))

        return contents if contents else [TextContent(type="text", text="Command executed successfully")]

    

class MCPBashTool(MCPTool):
    """
    MCP implementation copying BashTool20250124 from anthropic_tools.
    Allows running bash commands with MCP protocol.
    """
    def __init__(self):
        self._session = None
        super().__init__(
            name="bash",
            description="Run bash commands. You can execute shell commands and get their output. Use 'restart: true' to start a new shell session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "restart": {
                        "type": "boolean",
                        "description": "Whether to restart the bash session",
                        "default": False
                    }
                },
                "required": []
            }
        )

    async def execute(self, arguments: dict[str, Any]) -> Sequence[TextContent]:
        """Execute bash command - copied from BashTool20250124"""
        command = arguments.get("command")
        restart = arguments.get("restart", False)

        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return [TextContent(type="text", text="tool has been restarted.")]

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ValueError("no command provided.")


# Copied from anthropic_tools/edit.py with attribution
Command_20250429 = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
]
SNIPPET_LINES: int = 4


class MCPEditTool(MCPTool):
    """
    MCP implementation copying EditTool20250429 from anthropic_tools.
    Allows viewing, creating, editing files with MCP protocol.
    """

    _file_history: dict[Path, list[str]]

    def __init__(self):
        self._file_history = defaultdict(list)
        super().__init__(
            name="str_replace_editor",
            description="A tool for viewing, creating, and editing files. Supports viewing file contents, creating new files, replacing strings in files, and inserting text at specific lines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert"],
                        "description": "The operation to perform on the file"
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to operate on"
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Content for file creation (required for create command)"
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Range of lines to view [start, end] (for view command)"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to replace (required for str_replace command)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "String to replace with (for str_replace and insert commands)"
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number to insert at (required for insert command)"
                    }
                },
                "required": ["command", "path"]
            }
        )

    async def execute(self, arguments: dict[str, Any]) -> Sequence[TextContent]:
        """Execute file editing operation - copied from EditTool20250429"""
        command = arguments.get("command")
        path = arguments.get("path")
        file_text = arguments.get("file_text")
        view_range = arguments.get("view_range")
        old_str = arguments.get("old_str")
        new_str = arguments.get("new_str")
        insert_line = arguments.get("insert_line")

        _path = Path(path)
        self.validate_path(command, _path)
        if command == "view":
            return await self.view(_path, view_range)
        elif command == "create":
            if file_text is None:
                raise ValueError("Parameter `file_text` is required for command: create")
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            return [TextContent(type="text", text=f"File created successfully at: {_path}")]
        elif command == "str_replace":
            if old_str is None:
                raise ValueError("Parameter `old_str` is required for command: str_replace")
            return self.str_replace(_path, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                raise ValueError("Parameter `insert_line` is required for command: insert")
            if new_str is None:
                raise ValueError("Parameter `new_str` is required for command: insert")
            return self.insert(_path, insert_line, new_str)
        # Note: undo_edit command was removed in this version
        raise ValueError(
            f'Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command_20250429))}'
        )

    def validate_path(self, command: str, path: Path):
        """
        Check that the path/command combination is valid.
        """
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ValueError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ValueError(
                f"The path {path} does not exist. Please provide a valid path."
            )
        if path.exists() and command == "create":
            raise ValueError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        # Check if the path points to a directory
        if path.is_dir():
            if command != "view":
                raise ValueError(
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

    async def view(self, path: Path, view_range: list[int] | None = None) -> Sequence[TextContent]:
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ValueError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            _, stdout, stderr = await run(
                rf"find {path} -maxdepth 2 -not -path '*/\.*'"
            )
            contents = []
            if stdout or stderr:
                if not stderr:
                    stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                contents.append(TextContent(type="text", text=stdout))
                if stderr:
                    contents.append(TextContent(type="text", text=f"Error: {stderr}"))
            return contents

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ValueError(
                    "Invalid `view_range`. It should be a list of two integers."
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ValueError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ValueError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ValueError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return [TextContent(
            type="text",
            text=self._make_output(file_content, str(path), init_line=init_line)
        )]

    def str_replace(self, path: Path, old_str: str, new_str: str | None) -> Sequence[TextContent]:
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ValueError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Save the content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return [TextContent(type="text", text=success_msg)]

    def insert(self, path: Path, insert_line: int, new_str: str) -> Sequence[TextContent]:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ValueError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return [TextContent(type="text", text=success_msg)]

    # Note: undo_edit method is not implemented in this version as it was removed

    def read_file(self, path: Path) -> str:
        """Read the content of a file from a given path; raise an exception if an error occurs."""
        try:
            return path.read_text()
        except Exception as e:
            raise RuntimeError(f"Ran into {e} while trying to read {path}") from e

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise an exception if an error occurs."""
        try:
            path.write_text(file)
        except Exception as e:
            raise RuntimeError(f"Ran into {e} while trying to write to {path}") from e

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
