import asyncio
import os
from typing import Any, Sequence
from mcp.types import TextContent
from pydantic import PrivateAttr
from .shared import MCPTool
import sys
import shlex
from typing import Dict

# if on macos we infer we are local and use the current directory
if sys.platform == "darwin":
    automount_path = os.path.dirname(os.path.abspath(os.getcwd()))
    automount_path = os.path.join(automount_path, "sandbox")
    os.makedirs(automount_path, exist_ok=True)
else:
    automount_path = "/root/sandbox"
    os.makedirs(automount_path, exist_ok=True)

# Copied from anthropic_tools/bash.py with attribution
class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self, automount_path: str = automount_path):
        print("tool is using automount path:", automount_path)
        self._started = False
        self._timed_out = False
        self._automount_path = automount_path

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            cwd=self._automount_path,
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


class _ModalBashSession:
    """A stateful bash session inside a Modal Sandbox, mirroring _BashSession."""

    _started: bool
    _process: Any

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self, sandbox: Any, root: str = "/root/sandbox"):
        self._sandbox = sandbox
        self._root = root
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return
        # Start a long-lived interactive bash in the sandbox rooted at self._root
        # Use 'bash -lc' to run an initial cd and then a login shell
        self._process = self._sandbox.exec(
            "bash",
            "-lc",
            f"cd {shlex.quote(self._root)} && {self.command}",
            bufsize=1,
        )
        self._started = True

    def stop(self):
        if not self._started:
            raise RuntimeError("Session has not started.")
        try:
            # Try gracefully exiting the shell
            if getattr(self._process, "stdin", None) is not None:
                try:
                    self._process.stdin.write("exit\n")
                    self._process.stdin.drain()
                except Exception:
                    pass
            # Best-effort: if kill() exists, use it
            if hasattr(self._process, "kill"):
                try:
                    self._process.kill()
                except Exception:
                    pass
        finally:
            self._started = False

    def _run_blocking(self, command: str) -> Sequence[TextContent]:
        if not self._started:
            raise RuntimeError("Session has not started.")
        # Send command with sentinel
        self._process.stdin.write(command + f"; echo '{self._sentinel}'\n")
        self._process.stdin.drain()

        # Read until sentinel
        out_chunks: list[str] = []
        sentinel = self._sentinel
        for line in self._process.stdout:
            s = line.decode() if isinstance(line, (bytes, bytearray)) else str(line)
            if sentinel in s:
                s = s.split(sentinel)[0]
                if s:
                    out_chunks.append(s)
                break
            out_chunks.append(s)

        output = "".join(out_chunks).rstrip("\n")
        contents: list[TextContent] = []
        if output:
            contents.append(TextContent(type="text", text=output))
            
        return contents or [TextContent(type="text", text="Command executed successfully")]

    async def run(self, command: str) -> Sequence[TextContent]:
        # Wrap blocking read/write in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_blocking, command)

    

class MCPBashTool(MCPTool):
    """
    MCP implementation copying BashTool20250124 from anthropic_tools.
    Allows running bash commands with MCP protocol.
    """
    # Use PrivateAttr so Pydantic BaseModel allows this private state
    _session: Any = PrivateAttr(default=None)
    _sandbox: Any = PrivateAttr(default=None)
    _modal_session: Any = PrivateAttr(default=None)
    _jobs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, automount_path: str = automount_path, sandbox: Any | None = None, sandbox_root: str = "/root/sandbox"):
        super().__init__(
            name="bash",
            description=f"""
            Run bash commands. You can execute shell commands and get their output. 
            Use 'restart: true' to start a new shell session.
            Runs locally by default at {automount_path}. If a Modal Sandbox is provided at construction time, commands run inside it under {sandbox_root}.
            Background jobs: set 'background: true' with a 'name' to start a job whose logs are written under .mcp_logs. Use 'peek: true, name, lines' to view logs, 'poll: true, name' to get RUNNING/STOPPED, 'list_jobs: true' to list jobs, and 'stop: true, name' to stop.
            """,
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
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run the command as a background job and return immediately",
                        "default": False
                    },
                    "name": {
                        "type": "string",
                        "description": "Job name for background, peek, or stop"
                    },
                    "peek": {
                        "type": "boolean",
                        "description": "Peek logs for a named background job",
                        "default": False
                    },
                    "stop": {
                        "type": "boolean",
                        "description": "Stop a named background job",
                        "default": False
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of log lines to show when peeking",
                        "default": 100
                    },
                    "list_jobs": {
                        "type": "boolean",
                        "description": "List background jobs tracked by this session",
                        "default": False
                    },
                    "poll": {
                        "type": "boolean",
                        "description": "Poll status (RUNNING/STOPPED) for a named background job",
                        "default": False
                    }
                },
                "required": []
            }
        )
        self._sandbox = sandbox
        self._sandbox_root = sandbox_root
        self._automount_path = automount_path

    async def execute(self, arguments: dict[str, Any]) -> Sequence[TextContent]:
        """Execute bash command - copied from BashTool20250124"""
        command = arguments.get("command")
        restart = arguments.get("restart", False)
        background = arguments.get("background", False)
        peek = arguments.get("peek", False)
        stop = arguments.get("stop", False)
        name = arguments.get("name")
        lines = int(arguments.get("lines", 100))
        list_jobs = arguments.get("list_jobs", False)
        poll = arguments.get("poll", False)

        # If a sandbox was provided, run commands inside Modal in a stateful session
        if self._sandbox is not None:
            if restart:
                if self._modal_session:
                    self._modal_session.stop()
                self._modal_session = _ModalBashSession(self._sandbox, root=self._sandbox_root)
                await self._modal_session.start()
                return [TextContent(type="text", text="tool has been restarted.")]

            if self._modal_session is None:
                self._modal_session = _ModalBashSession(self._sandbox, root=self._sandbox_root)
                await self._modal_session.start()

            # Handle background/list/peek/stop/poll controls (Modal)
            logs_dir = f"{self._sandbox_root}/.mcp_logs"

            if list_jobs:
                if not self._jobs:
                    return [TextContent(type="text", text="No background jobs.")]
                lines_out: list[str] = []
                for jname, job in self._jobs.items():
                    pid = job.get("pid", -1)
                    logf = job.get("log", "")
                    out = await self._modal_session.run(f"kill -0 {pid} >/dev/null 2>&1; echo $?")
                    text = "\n".join(c.text for c in out if hasattr(c, 'text')).strip()
                    code = text.splitlines()[-1] if text else "1"
                    status = "RUNNING" if code == "0" else "STOPPED"
                    lines_out.append(f"- {jname}: {status} (pid {pid}) log: {logf}")
                return [TextContent(type="text", text="\n".join(lines_out))]

            if background:
                if not command:
                    raise ValueError("background requires a 'command'")
                if not name:
                    raise ValueError("background requires a 'name'")
                log_file = f"{logs_dir}/{name}.log"
                start_cmd = (
                    f"mkdir -p {shlex.quote(logs_dir)} && "
                    f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_file)} 2>&1 & echo $!"
                )
                out = await self._modal_session.run(start_cmd)
                # Combine output texts
                text = "\n".join(c.text for c in out if hasattr(c, 'text'))
                pid_line = text.strip().splitlines()[-1] if text.strip() else ""
                try:
                    pid = int(pid_line)
                except Exception:
                    pid = -1
                self._jobs[name] = {"pid": pid, "log": log_file}
                msg = (
                    f"Started job '{name}' pid {pid}. Logs: {log_file}.\n"
                    "This process is running in the background. You can:\n"
                    f"- peek logs: {{'peek': true, 'name': '{name}', 'lines': 100}}\n"
                    f"- poll status: {{'poll': true, 'name': '{name}'}}\n"
                    f"- stop job: {{'stop': true, 'name': '{name}'}}\n"
                    f"- list jobs: {{'list_jobs': true}}"
                )
                return [TextContent(type="text", text=msg)]

            if peek:
                if not name:
                    raise ValueError("peek requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                out = await self._modal_session.run(f"tail -n {lines} {shlex.quote(job['log'])}")
                return out

            if stop:
                if not name:
                    raise ValueError("stop requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                await self._modal_session.run(f"kill {job['pid']} || true")
                return [TextContent(type="text", text=f"stopped job '{name}' (pid {job['pid']})")]

            if poll:
                if not name:
                    raise ValueError("poll requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                out = await self._modal_session.run(f"kill -0 {job['pid']} >/dev/null 2>&1; echo $?")
                text = "\n".join(c.text for c in out if hasattr(c, 'text')).strip()
                code = text.splitlines()[-1] if text else "1"
                status = "RUNNING" if code == "0" else "STOPPED"
                return [TextContent(type="text", text=f"{name}: {status} (pid {job['pid']})")]

            if command is not None:
                return await self._modal_session.run(command)
            raise ValueError("no command provided.")

        # Local session mode
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession(automount_path=self._automount_path)
            await self._session.start()
            return [TextContent(type="text", text="tool has been restarted.")]

        if self._session is None:
            self._session = _BashSession(automount_path=self._automount_path)
            await self._session.start()

        # Background/list/peek/stop/poll for local mode
        logs_dir = os.path.join(self._automount_path, ".mcp_logs")
        if list_jobs:
            if not self._jobs:
                return [TextContent(type="text", text="No background jobs.")]
            lines_out: list[str] = []
            for jname, job in self._jobs.items():
                pid = job.get("pid", -1)
                logf = job.get("log", "")
                out = await self._session.run(f"kill -0 {pid} >/dev/null 2>&1; echo $?")
                text = "\n".join(c.text for c in out if hasattr(c, 'text')).strip()
                code = text.splitlines()[-1] if text else "1"
                status = "RUNNING" if code == "0" else "STOPPED"
                lines_out.append(f"- {jname}: {status} (pid {pid}) log: {logf}")
            return [TextContent(type="text", text="\n".join(lines_out))]
        if background:
            if not command:
                raise ValueError("background requires a 'command'")
            if not name:
                raise ValueError("background requires a 'name'")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"{name}.log")
            start_cmd = (
                f"mkdir -p {shlex.quote(logs_dir)} && "
                f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_file)} 2>&1 & echo $!"
            )
            out = await self._session.run(start_cmd)
            text = "\n".join(c.text for c in out if hasattr(c, 'text'))
            pid_line = text.strip().splitlines()[-1] if text.strip() else ""
            try:
                pid = int(pid_line)
            except Exception:
                pid = -1
            self._jobs[name] = {"pid": pid, "log": log_file}
            msg = (
                f"Started job '{name}' pid {pid}. Logs: {log_file}.\n"
                "This process is running in the background. You can:\n"
                f"- peek logs: {{'peek': true, 'name': '{name}', 'lines': 100}}\n"
                f"- poll status: {{'poll': true, 'name': '{name}'}}\n"
                f"- stop job: {{'stop': true, 'name': '{name}'}}\n"
                f"- list jobs: {{'list_jobs': true}}"
            )
            return [TextContent(type="text", text=msg)]

        if peek:
            if not name:
                raise ValueError("peek requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            out = await self._session.run(f"tail -n {lines} {shlex.quote(job['log'])}")
            return out

        if stop:
            if not name:
                raise ValueError("stop requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            await self._session.run(f"kill {job['pid']} || true")
            return [TextContent(type="text", text=f"stopped job '{name}' (pid {job['pid']})")]

        if poll:
            if not name:
                raise ValueError("poll requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            out = await self._session.run(f"kill -0 {job['pid']} >/dev/null 2>&1; echo $?")
            text = "\n".join(c.text for c in out if hasattr(c, 'text')).strip()
            code = text.splitlines()[-1] if text else "1"
            status = "RUNNING" if code == "0" else "STOPPED"
            return [TextContent(type="text", text=f"{name}: {status} (pid {job['pid']})")]

        if command is not None:
            return await self._session.run(command)

        raise ValueError("no command provided.")
