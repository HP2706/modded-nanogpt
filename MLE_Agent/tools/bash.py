import asyncio
import modal
import os
from typing import Any, Annotated
from pydantic import Field
from fastmcp import Context
import sys
import datetime
import shutil
from pathlib import Path
from typing import Dict
from modal.stream_type import StreamType
from .shared import LazySandBox
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

    def __init__(
        self, 
        automount_path: str = automount_path, 
        run_dir: str | None = None, 
        file_name: str | None = None
    ):
        self._started = False
        self._timed_out = False
        self._automount_path = automount_path
        self._file_name = f'environments/{file_name}'
        self._run_dir = run_dir
    
    async def start(self):
        if self._started:
            return

        # Ensure dated run directory exists and copy file_name if present
        os.makedirs(self._run_dir, exist_ok=True)
        for src in [
            Path.cwd() / self._file_name,
            Path(self._automount_path).parent / self._file_name,
        ]:
            if src.is_file():
                try:
                    shutil.copy2(src, Path(self._run_dir) / self._file_name)
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

    async def run(self, command: str) -> str:
        """Execute a command in the bash shell."""
        if not self._started:
            return "tool must be restarted\nSession has not started."
        if self._process.returncode is not None:
            self._process.wait()
            return (
                f"Error: bash has exited with "
                f"returncode {self._process.returncode}"
                f"stdout {self._process.stdout.read()}"
                f"stderr {self._process.stderr.read()}"
            )

        if self._timed_out:
            return f"Error: bash has timed out after {self._timeout} seconds and must be restarted"
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

        result_parts = []
        if output:
            result_parts.append(output)
        if error:
            result_parts.append(f"Error: {error}")
        return "\n".join(result_parts) if result_parts else "Command executed successfully"

class _ModalBashSession:
    """A stateful bash session inside a Modal Sandbox, mirroring _BashSession."""

    _started: bool
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds

    def __init__(
        self,
        sandbox: Any,
        root: str = "/root/sandbox",
        run_dir: str | None = None,
        file_name: str | None = None,
    ):
        self._sandbox = sandbox
        self._root = root
        self._started = False
        self._timed_out = False
        
        self._file_name = file_name
        self._run_dir = run_dir
        self._prefix_cmd = f'cd {self._run_dir}'
        self._started = False
        
    async def start(self):
        if self._started:
            return
        
        setup_cmd = (
            f"mkdir -p {self._run_dir} && "
            f"cp -n /root/sandbox/{self._file_name} {(self._run_dir)} || true"
        ) # cp -n means copy only if the file does not exist

        _p = self._sandbox.exec(
            "bash",
            "-c",
            setup_cmd,
            stdout=StreamType.PIPE,
            stderr=StreamType.PIPE,
        )
        _p.wait()

        self._started = True
        print(f"setup_cmd: {setup_cmd} _p: {_p.stdout.read()} {_p.stderr.read()}")

    async def run(
        self, 
        command: str, 
        blocking: bool = True,
    ) -> str:
        # Wrap blocking read/write in a thread to avoid blocking the event loop
        if not self._started:
            await self.start()
            
        _p = self._sandbox.exec(
            "bash",
            "-c",
            f"{self._prefix_cmd} && {command}",
        )
        
        if blocking:
            _p.wait()
            stdout = _p.stdout.read()
            stderr = _p.stderr.read()
            return f"stdout: {stdout}\nstderr: {stderr} returncode: {_p.returncode}"
    
class BashContainer:
    """
    Stateful bash container that supports local and Modal Sandbox execution,
    background jobs, and session restart. 
    """
    _session: _ModalBashSession | _BashSession
    def __init__(
        self,
        sandbox: LazySandBox | None = None,
        automount_path: str = '/root/sandbox',
        run_dir: str | None = None,
        file_name: str | None = None,
    ):
        self._sandbox = sandbox
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._sandbox_root = automount_path
        self._automount_path = automount_path
        if run_dir:
            self._run_dir = run_dir
        else:
            self._run_dir = os.path.join(self._automount_path, "runs", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S"))

        self._file_name = file_name
        if sandbox is not None:
            self._session = _ModalBashSession(sandbox, root=automount_path, run_dir=self._run_dir, file_name=file_name)
        else:
            self._session = _BashSession(automount_path=automount_path, run_dir=self._run_dir, file_name=file_name)
        
        # spawn in a thread
        asyncio.create_task(self._session.start())
        #self._session.start()
        
    def ensure_cwd(self) -> str:
        """Ensure a session is started and return its working directory."""
        self._session.start()
        return self._session._run_dir

    async def restart_session(
        self,
        ctx: Annotated[Context | None, Field(description="Optional context for debugging")] = None,
        run_dir: Annotated[str | None, Field(description="Optional existing run directory to reuse instead of creating a new one")] = None,
    ) -> str:
        """Restart the bash session. If run_dir is provided, reuse it; otherwise create a new timestamped run dir."""
        ctx.debug("restarting bash session") if ctx else None
        # Stop isn't strictly necessary since we start a fresh session object
        # Create new session with optional run_dir
        if isinstance(self._session, _ModalBashSession):
            self._session = _ModalBashSession(self._sandbox, root=self._sandbox_root, run_dir=run_dir, file_name=self._file_name)
        else:
            self._session = _BashSession(automount_path=self._automount_path, run_dir=run_dir, file_name=self._file_name)
        await self._session.start()
        return f"tool has been restarted. cwd: {self._session._run_dir}"

    async def list_jobs(self) -> Annotated[str, Field(description="List of all background jobs with their status")]:
        """List all background jobs tracked by this session."""
        if not self._jobs:
            return "No background jobs."
        lines_out: list[str] = []
        for jname, job in self._jobs.items():
            pid = job.get("pid", -1)
            out = await self._session.run(f"kill -0 {pid} >/dev/null 2>&1; echo $?")
            text = out.strip()
            code = text.splitlines()[-1] if text else "1"
            status = "RUNNING" if code == "0" else "STOPPED"
            lines_out.append(f"- {jname}: {status} (pid {pid}) log: {job.get('log','')}")
        return "\n".join(lines_out)

    async def run_command_background(
        self,
        command: Annotated[str, Field(
        description=(
            "The bash command to run in background,"
            "very useful if you want to run a long job like a training run, etc."
        ))],
        name: Annotated[str, Field(description="Unique name for the background job")]
    ) -> Annotated[str, Field(description="Job startup confirmation with management instructions")]:
        """Run a bash command as a background job and return immediately."""
        if not command:
            raise ValueError("background requires a 'command'")
        if not name:
            raise ValueError("background requires a 'name'")

        logs_dir = ".mcp_logs"

        # create logs dir if it doesn't exist
        text = await self._session.run(f"mkdir -p {logs_dir}")
        print('text post mkdir', text)
        log_file = f"{logs_dir}/{name}.log"

        logs_dir = f"{self._session._run_dir}/.mcp_logs"
        log_file = f"{logs_dir}/{name}.log"

        command = f"""
        nohup bash -lc 'echo $$ >> {log_file}.pid; {command}' | tee {log_file}
        """
        #nohup bash -lc 'echo $$ >> {log_file}.pid; your_command' | tee {log_file}
        # get pid by reading first line in tee log file
        _ = await self._session.run(command, blocking=False)
        await asyncio.sleep(3) # give enough time for the pid to be written to the log file

        pid_text = await self._session.run(f"sed -n '1p' {log_file}.pid")
        #pid_text = stdout: pid
        pid = int(pid_text.split(': ')[1].strip())
        self._jobs[name] = {"pid": pid, "log": log_file}

        return (
            f"Started job '{name}' pid {pid}. Logs: {log_file}.\n"
            "This process is running in the background. You can:\n"
            f"- if you want to peek the logs, you can read specific lines via sed.\n"
            f"- for example to read the first 100 lines, you can use: sed -n '1,100p' '{log_file}'"
            f"- poll status: {{'poll': true, 'name': '{name}'}}\n"
            f"- stop job: {{'stop': true, 'name': '{name}'}}\n"
            f"- list jobs: {{'list_jobs': true}}"
        )

    async def stop_job(
        self,
        name: Annotated[str, Field(description="Name of the background job to stop")]
    ) -> Annotated[str, Field(description="Confirmation that the job has been stopped")]:
        """Stop a named background job."""
        if not name:
            raise ValueError("stop requires a 'name'")
        
        job = self._jobs.get(name)
        if not job:
            raise ValueError(f"unknown job '{name}'")
        await self._session.run(f"kill {job['pid']} || true")
        return f"stopped job '{name}' (pid {job['pid']})"

    async def poll_job(
        self,
        name: Annotated[str, Field(description="Name of the background job to check status")]
    ) -> Annotated[str, Field(description="Current status of the background job")]:
        """Poll status (RUNNING/STOPPED) for a named background job."""
        if not name:
            raise ValueError("poll requires a 'name'")
        job = self._jobs.get(name)
        if not job:
            raise ValueError(f"unknown job '{name}'")
        out = await self._session.run(f"kill -0 {job['pid']} >/dev/null 2>&1; echo $?")
        text = out.strip()
        code = text.splitlines()[-1] if text else "1"
        status = "RUNNING" if code == "0" else "STOPPED"
        return f"{name}: {status} (pid {job['pid']})"

    async def run_command(
        self,
        command: Annotated[str, Field(description="The bash command to execute")]
    ) -> Annotated[str, Field(description="Output from the executed bash command")]:
        """Execute a regular bash command."""
        text = await self._session.run(command)
        # ensure it is not too long
        if len(text) > 10000:
            return text[:10000] + "..."
        return text

    async def stop_background_job(
        self,
        name: Annotated[str, Field(description="Name of the background job to stop")]
    ) -> Annotated[str, Field(description="Confirmation that the job has been stopped")]:
        """Stop a background job."""
        return await self.stop_job(name)

    async def poll_background_job(
        self,
        name: Annotated[str, Field(description="Name of the background job to check status")]
    ) -> Annotated[str, Field(description="Current status of the background job")]:
        """Check status of a background bash job."""
        return await self.poll_job(name)

    async def list_background_jobs(self) -> Annotated[str, Field(description="List of all background jobs with their status")]:
        """List all background bash jobs."""
        return await self.list_jobs()
