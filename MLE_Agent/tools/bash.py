import asyncio
import os
from typing import Any, Sequence, Annotated
from pydantic import Field
from mcp.types import TextContent
from fastmcp import Context
import sys
import shlex
import datetime
import shutil
from pathlib import Path
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
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = os.path.join(self._automount_path, "runs", ts)

    async def start(self):
        if self._started:
            return

        # Ensure dated run directory exists and copy modded-nanogpt.py if present
        os.makedirs(self._run_dir, exist_ok=True)
        for src in [
            Path.cwd() / "modded-nanogpt.py",
            Path(self._automount_path).parent / "modded-nanogpt.py",
        ]:
            if src.is_file():
                try:
                    shutil.copy2(src, Path(self._run_dir) / "modded-nanogpt.py")
                except Exception:
                    pass
                break

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            cwd=self._run_dir,
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
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = f"{self._root}/runs/{ts}"
        

    async def start(self):
        if self._started:
            return
        
        print(f"starting modal bash session in {self._run_dir}")
        
        
        
        # Prepare run directory and copy modded-nanogpt.py if present
        setup = (
            f"mkdir -p {shlex.quote(self._run_dir)} && "
            f"( [ -f /root/modded-nanogpt.py ] && cp /root/modded-nanogpt.py {shlex.quote(self._run_dir)}/ || true ) && "
            f"( [ -f {shlex.quote(self._root)}/modded-nanogpt.py ] && cp {shlex.quote(self._root)}/modded-nanogpt.py {shlex.quote(self._run_dir)}/ || true )"
        )
        
        # Start a long-lived interactive bash in the run directory
        self._process = self._sandbox.exec(
            "bash",
            "-lc",
            f"{setup} && cd {shlex.quote(self._run_dir)} && exec {self.command}",
            bufsize=1,
        )
        self._process.wait()
        print(f"setup: {setup}", "stdout:", self._process.stdout.read(), "stderr:", self._process.stderr.read())
        print(f"process: {self._process.returncode}")
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
    
    
    
class BashContainer:
    """
    Stateful bash container that supports local and Modal Sandbox execution,
    background jobs, and session restart. Register with FastMCP via mcp.tool(obj.bash).
    """

    def __init__(self, automount_path: str = automount_path, sandbox: Any | None = None, sandbox_root: str = "/root/sandbox"):
        self._session: Any | None = None
        self._sandbox: Any | None = sandbox
        self._modal_session: Any | None = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._sandbox_root = sandbox_root
        self._automount_path = automount_path

    async def ensure_cwd(self) -> str:
        """Ensure a session is started and return its working directory."""
        if self._sandbox is not None:
            if self._modal_session is None:
                self._modal_session = _ModalBashSession(self._sandbox, root=self._sandbox_root)
                await self._modal_session.start()
            return self._modal_session._run_dir
        else:
            if self._session is None:
                self._session = _BashSession(automount_path=self._automount_path)
                await self._session.start()
            return self._session._run_dir

    async def bash(
        self,
        command: Annotated[str | None, Field(description="The bash command to execute")]=None,
        restart: Annotated[bool, Field(description="Restart the bash session (new run dir)")]=False,
        background: Annotated[bool, Field(description="Run command as a background job and return immediately")]=False,
        name: Annotated[str | None, Field(description="Job name for background, peek, stop, or poll")]=None,
        peek: Annotated[bool, Field(description="Peek logs for a named background job")]=False,
        stop: Annotated[bool, Field(description="Stop a named background job")]=False,
        lines: Annotated[int, Field(description="Number of log lines to show when peeking", ge=1, le=10000)]=100,
        list_jobs: Annotated[bool, Field(description="List background jobs tracked by this session")]=False,
        poll: Annotated[bool, Field(description="Poll status (RUNNING/STOPPED) for a named background job")]=False,
        ctx : Context = None
    ) -> str:
        '''
        Args: 
            
        '''
        ctx.debug(f"bash command: {command}")
        # Modal sandboxed session
        if self._sandbox is not None:
            if restart:
                if self._modal_session:
                    self._modal_session.stop()
                self._modal_session = _ModalBashSession(self._sandbox, root=self._sandbox_root)
                await self._modal_session.start()
                return f"tool has been restarted. cwd: {self._modal_session._run_dir}"

            if self._modal_session is None:
                self._modal_session = _ModalBashSession(self._sandbox, root=self._sandbox_root)
                await self._modal_session.start()

            if (
                command is None and not background and not peek and not stop and not list_jobs and not poll
            ):
                return f"tool started. cwd: {self._modal_session._run_dir}"

            logs_dir = f"{self._modal_session._run_dir}/.mcp_logs"

            if list_jobs:
                if not self._jobs:
                    return "No background jobs."
                lines_out: list[str] = []
                for jname, job in self._jobs.items():
                    pid = job.get("pid", -1)
                    out = await self._modal_session.run(f"kill -0 {pid} >/dev/null 2>&1; echo $?")
                    text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', '')).strip()
                    code = text.splitlines()[-1] if text else "1"
                    status = "RUNNING" if code == "0" else "STOPPED"
                    lines_out.append(f"- {jname}: {status} (pid {pid}) log: {job.get('log','')}")
                return "\n".join(lines_out)

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
                text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))
                pid_line = text.strip().splitlines()[-1] if text.strip() else ""
                try:
                    pid = int(pid_line)
                except Exception:
                    pid = -1
                self._jobs[name] = {"pid": pid, "log": log_file}
                return (
                    f"Started job '{name}' pid {pid}. Logs: {log_file}.\n"
                    "This process is running in the background. You can:\n"
                    f"- peek logs: {{'peek': true, 'name': '{name}', 'lines': 100}}\n"
                    f"- poll status: {{'poll': true, 'name': '{name}'}}\n"
                    f"- stop job: {{'stop': true, 'name': '{name}'}}\n"
                    f"- list jobs: {{'list_jobs': true}}"
                )

            if peek:
                if not name:
                    raise ValueError("peek requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                log = shlex.quote(job['log'])
                out = await self._modal_session.run(
                    f"if [ -f {log} ]; then tail -n {lines} {log}; else echo 'no logs yet for {name}'; fi"
                )
                return "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))

            if stop:
                if not name:
                    raise ValueError("stop requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                await self._modal_session.run(f"kill {job['pid']} || true")
                return f"stopped job '{name}' (pid {job['pid']})"

            if poll:
                if not name:
                    raise ValueError("poll requires a 'name'")
                job = self._jobs.get(name)
                if not job:
                    raise ValueError(f"unknown job '{name}'")
                out = await self._modal_session.run(f"kill -0 {job['pid']} >/dev/null 2>&1; echo $?")
                text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', '')).strip()
                code = text.splitlines()[-1] if text else "1"
                status = "RUNNING" if code == "0" else "STOPPED"
                return f"{name}: {status} (pid {job['pid']})"

            if command is not None:
                out = await self._modal_session.run(command)
                return "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))

            raise ValueError("no command provided.")

        # Local session mode
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession(automount_path=self._automount_path)
            await self._session.start()
            return f"tool has been restarted. cwd: {self._session._run_dir}"

        if self._session is None:
            self._session = _BashSession(automount_path=self._automount_path)
            await self._session.start()

        if (
            command is None and not background and not peek and not stop and not list_jobs and not poll
        ):
            return f"tool started. cwd: {self._session._run_dir}"

        logs_dir = os.path.join(self._session._run_dir, ".mcp_logs")

        if list_jobs:
            if not self._jobs:
                return "No background jobs."
            lines_out: list[str] = []
            for jname, job in self._jobs.items():
                pid = job.get("pid", -1)
                out = await self._session.run(f"kill -0 {pid} >/dev/null 2>&1; echo $?")
                text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', '')).strip()
                code = text.splitlines()[-1] if text else "1"
                status = "RUNNING" if code == "0" else "STOPPED"
                lines_out.append(f"- {jname}: {status} (pid {pid}) log: {job.get('log','')}")
            return "\n".join(lines_out)

        if background:
            if not command:
                raise ValueError("background requires a 'command'")
            if not name:
                raise ValueError("background requires a 'name'")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"{name}.log")
            out = await self._session.run(
                f"mkdir -p {shlex.quote(logs_dir)} && nohup bash -lc {shlex.quote(command)} > {shlex.quote(log_file)} 2>&1 & echo $!"
            )
            text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))
            pid_line = text.strip().splitlines()[-1] if text.strip() else ""
            try:
                pid = int(pid_line)
            except Exception:
                pid = -1
            self._jobs[name] = {"pid": pid, "log": log_file}
            return (
                f"Started job '{name}' pid {pid}. Logs: {log_file}.\n"
                "This process is running in the background. You can:\n"
                f"- peek logs: {{'peek': true, 'name': '{name}', 'lines': 100}}\n"
                f"- poll status: {{'poll': true, 'name': '{name}'}}\n"
                f"- stop job: {{'stop': true, 'name': '{name}'}}\n"
                f"- list jobs: {{'list_jobs': true}}"
            )

        if peek:
            if not name:
                raise ValueError("peek requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            log = shlex.quote(job['log'])
            out = await self._session.run(
                f"if [ -f {log} ]; then tail -n {lines} {log}; else echo 'no logs yet for {name}'; fi"
            )
            return "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))

        if stop:
            if not name:
                raise ValueError("stop requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            await self._session.run(f"kill {job['pid']} || true")
            return f"stopped job '{name}' (pid {job['pid']})"

        if poll:
            if not name:
                raise ValueError("poll requires a 'name'")
            job = self._jobs.get(name)
            if not job:
                raise ValueError(f"unknown job '{name}'")
            out = await self._session.run(f"kill -0 {job['pid']} >/dev/null 2>&1; echo $?")
            text = "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', '')).strip()
            code = text.splitlines()[-1] if text else "1"
            status = "RUNNING" if code == "0" else "STOPPED"
            return f"{name}: {status} (pid {job['pid']})"

        if command is not None:
            out = await self._session.run(command)
            return "\n".join(getattr(c, 'text', '') for c in out if getattr(c, 'text', ''))

        raise ValueError("no command provided.")
