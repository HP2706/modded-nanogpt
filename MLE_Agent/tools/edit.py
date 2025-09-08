from typing import Any, Literal, Annotated
from pydantic import Field
from .shared import maybe_truncate, run
from pathlib import Path
from collections import defaultdict
from typing import get_args
import shlex
import os
import sys

# if on macOS we infer we are local and use the current directory's parent/sandbox
if sys.platform == "darwin":
    automount_path = os.path.dirname(os.path.abspath(os.getcwd()))
    automount_path = os.path.join(automount_path, "sandbox")
    os.makedirs(automount_path, exist_ok=True)
else:
    automount_path = "/root/sandbox"
    os.makedirs(automount_path, exist_ok=True)

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

    def __init__(self, sandbox: Any | None = None, sandbox_root: str = "/root/sandbox", automount: str = automount_path):
        # Track history per file
        self._file_history: dict[Path, list[str]] = defaultdict(list)
        self._sandbox: Any | None = sandbox
        self._sandbox_root: str = sandbox_root
        # local automount path (used when not in sandbox)
        self._automount_path: str = automount

    async def view(
        self, 
        path: Annotated[Path, 
            Field(description="Path to the file or directory to view")
        ], 
        view_range: Annotated[list[int] | None, 
            Field(description="Range of lines to view in a specific file, [start, end]; use -1 as end. Not allowed for directories.")
        ] = None
    ) -> str:
        """Implement the view command; returns formatted text."""
        self._validate_path("view", path)
        # Directory listing
        if self._is_dir(path):
            if view_range:
                raise ValueError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            if self._sandbox is None:
                # Map /root/sandbox to local automount path if on macOS
                resolved = self._resolve_local_path(path)
                _, stdout, stderr = await run(
                    rf"find {resolved} -maxdepth 2 -not -path '*/\.*'"
                )
                msg = ""
                if stdout and not stderr:
                    msg = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                elif stderr:
                    msg = f"Error: {stderr}"
                return msg or ""
            else:
                resolved = self._resolve_sandbox_path(path)
                out = self._sb_run(rf"find {shlex.quote(resolved)} -maxdepth 2 -not -path '*/\.*'")
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

    def str_replace(
        self, 
        path: Annotated[Path, Field(description="Path to the file to perform string replacement in")] , 
        old_str: Annotated[str, Field(description="String to replace")] , 
        new_str: Annotated[str | None, Field(description="Replacement string (use empty string to delete)")] = None
    ) -> str:
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        self._validate_path("str_replace", path)
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ValueError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}"
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of old_str"
                f"`{old_str}` in lines {lines}. Please ensure it is unique"
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

    def insert(
        self, 
        path: Annotated[Path, Field(description="Path to the file to insert into")] ,   
        insert_line: Annotated[int, Field(description="Line number where to insert the string")] , 
        new_str: Annotated[str, Field(description="String to insert")]
    ) -> str:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        self._validate_path("insert", path)
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ValueError(
                f"Invalid `insert_line` parameter: {insert_line}."
                f"It should be within the range of lines of the file: {[0, n_lines_file]}"
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
    
    def read_file(
        self, 
        path: Annotated[Path, Field(description="Path to the file to read")]
    ) -> str:
        """Read the content of a file from a given path; raise an exception if an error occurs."""
        self._validate_path("read", path)
        if self._sandbox is None:
            try:
                resolved = self._resolve_local_path(path)
                return resolved.read_text()
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to read {path}") from e
        else:
            try:
                resolved = self._resolve_sandbox_path(path)
                return self._sb_run(f"cat {shlex.quote(resolved)}")
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to read {path} in sandbox") from e

    def write_file(
        self, 
        path: Annotated[Path, Field(description="Path to the file to write")] , 
        file: str
    ):
        """Write the content of a file to a given path; raise an exception if an error occurs."""
        self._validate_path("write", path)
        if self._sandbox is None:
            try:
                resolved = self._resolve_local_path(path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(file)
            except Exception as e:
                raise RuntimeError(f"Ran into {e} while trying to write to {path}") from e
        else:
            try:
                resolved = self._resolve_sandbox_path(path)
                parent_dir = os.path.dirname(resolved)
                # Use a single-quoted heredoc and ensure parent directories exist
                heredoc = (
                    f"mkdir -p {shlex.quote(parent_dir)} && "
                    f"cat > {shlex.quote(resolved)} << 'EOF'\n{file}\nEOF\n"
                )
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
        
    def _validate_path(self, command: str, path: Path):
        """
        Check that the path/command combination is valid.
        """
        # Resolve path for local automount checks, but keep original in messages
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
            resolved = self._resolve_local_path(path)
            return resolved.exists()
        resolved = self._resolve_sandbox_path(path)
        out = self._sb_run(f"if test -e {shlex.quote(resolved)}; then echo yes; else echo no; fi")
        return out.strip().endswith("yes")

    def _is_dir(self, path: Path) -> bool:
        if self._sandbox is None:
            resolved = self._resolve_local_path(path)
            return resolved.is_dir()
        resolved = self._resolve_sandbox_path(path)
        out = self._sb_run(f"if test -d {shlex.quote(resolved)}; then echo yes; else echo no; fi")
        return out.strip().endswith("yes")

    def _resolve_local_path(self, path: Path) -> Path:
        """Resolve any given path (absolute or relative) into the local automount root.
        All operations are confined under self._automount_path.
        """
        p_str = str(path)
        # If already under automount, keep it
        if p_str.startswith(self._automount_path):
            return Path(p_str)
        # If user specified /root/sandbox prefix, strip it
        if p_str.startswith("/root/sandbox"):
            suffix = p_str[len("/root/sandbox"):].lstrip("/")
        else:
            suffix = p_str.lstrip("/") if path.is_absolute() else p_str
        return Path(self._automount_path) / suffix

    def _resolve_sandbox_path(self, path: Path) -> str:
        """Resolve any given path (absolute or relative) into the sandbox automount root.
        All operations are confined under self._sandbox_root in sandbox mode.
        """
        p_str = str(path)
        if p_str.startswith(self._sandbox_root):
            return p_str
        suffix = p_str.lstrip("/") if path.is_absolute() else p_str
        return os.path.join(self._sandbox_root, suffix)
