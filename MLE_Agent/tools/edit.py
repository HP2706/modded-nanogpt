from typing import Literal, Annotated
from pydantic import Field
from .shared import maybe_truncate, LazySandBox
from pathlib import Path
from collections import defaultdict
import shlex
import os
import sys
import subprocess

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

    def __init__(
        self, 
        sandbox: LazySandBox | None = None, 
        automount_path: str = automount_path
    ):
        # Track history per file
        self._file_history: dict[Path, list[str]] = defaultdict(list)
        self._sandbox = sandbox
        # local automount path (used when not in sandbox)
        self._automount_path: str = automount_path

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
        path = self._resolve_local_path(path)
        stdout = self._list_dir(path)
        print(path, f"self._list_dir(path): {stdout}")
        # Directory listing
        if self.dir_or_file_exists(path, is_dir=True):
            if view_range:
                raise ValueError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            if self._sandbox is None:
                # Map /root/sandbox to local automount path if on macOS

                sub_cmd = rf"find {path} -maxdepth 2 -not -path '*/\.*'"
                
                output = subprocess.run(sub_cmd, shell=True, capture_output=True, text=True)
                stdout = output.stdout
                stderr = output.stderr
                
                msg = ""
                if stdout and not stderr:
                    msg = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                elif stderr:
                    msg = f"Error: {stderr}"
                return msg or ""
            else:
                p = await self._sandbox.exec(
                    'bash',
                    '-c',
                    rf"find {path} -maxdepth 2 -not -path '*/\.*'",
                )
                p.wait()
                stdout = p.stdout.read()

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
        path = self._resolve_local_path(path)
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
        path = self._resolve_local_path(path)
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
        path = self._resolve_local_path(path)
        resolved = self._resolve_local_path(path)
        if self._sandbox is None:
            return resolved.read_text()
        else:
            output = self._sandbox.exec('bash', '-c', f"cat {resolved}")
            output.wait()
            stdout = output.stdout.read()
            stderr = output.stderr.read()
            if stderr != "":
                print(f"error: {stderr}")
            return stdout

    def write_file(
        self, 
        path: Annotated[Path, Field(description="Path to the file to write")] , 
        file: str
    ) -> str:
        """Write the content of a file to a given path; raise an exception if an error occurs."""
        path = self._resolve_local_path(path)
        if self._sandbox is None:
            resolved = self._resolve_local_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(file)
        else:
            resolved = self._resolve_local_path(path)
            parent_dir = os.path.dirname(resolved)
            # Use a single-quoted heredoc and ensure parent directories exist
            heredoc = (
                f"mkdir -p {parent_dir} && "
                f"cat > {str(resolved)} << 'EOF'\n{file}\nEOF\n"
            )
            p = self._sandbox.exec('bash', '-c', heredoc)
            p.wait()
        
        return f"The file {path} has been written."
    
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
        
    def _list_dir(self, path: Path) -> str:
        path = self._resolve_local_path(path)
        if self._sandbox is None:
            return str(os.listdir(path))
        else:
            p = self._sandbox.exec('bash', '-c', f'ls {path}')
            p.wait()
            stdout = p.stdout.read()
            stderr = p.stderr.read()
            if stderr.strip() != "":
                print(f"error: {stderr.strip()}")
            return stdout

    # --- Helpers for Modal sandbox integration ---
    def dir_or_file_exists(self, path: Path, is_dir: bool) -> bool:
        path = self._resolve_local_path(path)
        if self._sandbox is None:
            if is_dir:
                return path.is_dir()
            else:
                return path.exists()
        else:
            if is_dir:
                test_flag = "d"
            else:
                test_flag = "e"
            out = self._sandbox.exec('bash', '-c', f"if test -{test_flag} {str(path)}; then echo yes; else echo no; fi")
            out.wait()
            stdout = out.stdout.read()
            stderr = out.stderr.read()
            if stderr.strip() != "":
                print(f"error: {stderr.strip()}")
            val = stdout.strip().endswith("yes")
            return val

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
        
        path = Path(self._automount_path) / suffix
        
        if self._sandbox is not None:
            p = self._sandbox.exec('bash', '-c', f'ls {path}')
            p.wait()
            stdout = p.stdout.read()
            stderr = p.stderr.read()
            print(f"error: {stderr.strip()}")
            print(f"stdout: {stdout}")
        
        return path

