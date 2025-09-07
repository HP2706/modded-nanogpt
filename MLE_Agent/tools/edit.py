from typing import Any, Literal, Annotated
from pydantic import Field
from .shared import maybe_truncate, run
from pathlib import Path
from collections import defaultdict
from typing import get_args
import shlex

# Copied from anthropic_tools/edit.py with attribution
Command_20250429 = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
]
SNIPPET_LINES: int = 4


class EditContainer:
    """Stateful file edit container supporting view, create, str_replace, and insert."""

    def __init__(self, sandbox: Any | None = None, sandbox_root: str = "/root/sandbox"):
        # Track history per file
        self._file_history: dict[Path, list[str]] = defaultdict(list)
        self._sandbox: Any | None = sandbox
        self._sandbox_root: str = sandbox_root

    async def str_replace_editor(
        self,
        command: Annotated[Literal["view", "create", "str_replace", "insert"], Field(description="The operation to perform: view, create, str_replace, or insert")],
        path: Annotated[str, Field(description="Absolute path to the file (or directory for view)")],
        file_text: Annotated[str | None, Field(description="File content to create (required for create)")]=None,
        view_range: Annotated[list[int] | None, Field(description="Range of lines to view [start, end]; use -1 as end")]=None,
        old_str: Annotated[str | None, Field(description="String to replace (required for str_replace)")]=None,
        new_str: Annotated[str | None, Field(description="Replacement string (for str_replace and insert)")]=None,
        insert_line: Annotated[int | None, Field(description="Line number to insert at (required for insert)")]=None,
    ) -> str:
        _path = Path(path)
        self.validate_path(command, _path)
        if command == "view":
            return await self._view(_path, view_range)
        if command == "create":
            if file_text is None:
                raise ValueError("Parameter `file_text` is required for command: create")
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            return f"File created successfully at: {_path}"
        if command == "str_replace":
            if old_str is None:
                raise ValueError("Parameter `old_str` is required for command: str_replace")
            return self._str_replace(_path, old_str, new_str)
        if command == "insert":
            if insert_line is None:
                raise ValueError("Parameter `insert_line` is required for command: insert")
            if new_str is None:
                raise ValueError("Parameter `new_str` is required for command: insert")
            return self._insert(_path, insert_line, new_str)
        raise ValueError(
            f"Unrecognized command {command}. Allowed: {', '.join(get_args(Command_20250429))}"
        )

    def validate_path(self, command: str, path: Path):
        """
        Check that the path/command combination is valid.
        """
        # Require absolute paths to avoid ambiguity.
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ValueError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )

        exists = self._exists(path)
        is_dir = self._is_dir(path) if exists else False

        if not exists and command != "create":
            raise ValueError(
                f"The path {path} does not exist. Please provide a valid path."
            )
        if exists and command == "create":
            raise ValueError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        if is_dir and command != "view":
            raise ValueError(
                f"The path {path} is a directory and only the `view` command can be used on directories"
            )

    async def _view(self, path: Path, view_range: list[int] | None = None) -> str:
        """Implement the view command; returns formatted text."""
        # Directory listing
        if self._is_dir(path):
            if view_range:
                raise ValueError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            if self._sandbox is None:
                _, stdout, stderr = await run(
                    rf"find {path} -maxdepth 2 -not -path '*/\.*'"
                )
                msg = ""
                if stdout and not stderr:
                    msg = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                elif stderr:
                    msg = f"Error: {stderr}"
                return msg or ""
            else:
                out = self._sb_run(rf"find {shlex.quote(str(path))} -maxdepth 2 -not -path '*/\.*'")
                stdout = out.strip()
                return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"

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

        return self._make_output(file_content, str(path), init_line=init_line)

    def _str_replace(self, path: Path, old_str: str, new_str: str | None) -> str:
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

        return success_msg

    def _insert(self, path: Path, insert_line: int, new_str: str) -> str:
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
        return success_msg

    # Note: undo_edit method is not implemented in this version as it was removed

    def read_file(self, path: Path) -> str:
        """Read the content of a file from a given path; raise an exception if an error occurs."""
        if self._sandbox is None:
            try:
                return path.read_text()
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to read {path}") from e
        else:
            try:
                return self._sb_run(f"cat {shlex.quote(str(path))}")
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to read {path} in sandbox") from e

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise an exception if an error occurs."""
        if self._sandbox is None:
            try:
                path.write_text(file)
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to write to {path}") from e
        else:
            try:
                # Use a single-quoted heredoc to avoid interpolation issues and ensure EOF is on its own line
                heredoc = f"cat > {shlex.quote(str(path))} << 'EOF'\n{file}\nEOF\n"
                self._sb_run(heredoc)
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to write to {path} in sandbox") from e

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

    # --- Helpers for Modal sandbox integration ---
    def _sb_run(self, cmd: str) -> str:
        """Run a command inside the provided sandbox and return stdout as text."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not configured")
        # Prefix with cd to sandbox root for consistency
        full_cmd = f"cd {shlex.quote(self._sandbox_root)} && {cmd}; echo '<<exit>>'"
        p = self._sandbox.exec("bash", "-lc", full_cmd, bufsize=1)
        sentinel = "<<exit>>"
        out_chunks: list[str] = []
        for line in p.stdout:
            s = line.decode() if isinstance(line, (bytes, bytearray)) else str(line)
            if sentinel in s:
                s = s.split(sentinel)[0]
                if s:
                    out_chunks.append(s)
                break
            out_chunks.append(s)
        p.wait()
        return "".join(out_chunks)

    def _exists(self, path: Path) -> bool:
        if self._sandbox is None:
            return path.exists()
        out = self._sb_run(f"if test -e {shlex.quote(str(path))}; then echo yes; else echo no; fi")
        return out.strip().endswith("yes")

    def _is_dir(self, path: Path) -> bool:
        if self._sandbox is None:
            return path.is_dir()
        out = self._sb_run(f"if test -d {shlex.quote(str(path))}; then echo yes; else echo no; fi")
        return out.strip().endswith("yes")
