from modal import Sandbox, Image
import modal
from modal.stream_type import StreamType
import shlex
import datetime
import os

def test_sandbox():


    vol = modal.Volume.from_name("mle-sandbox", create_if_missing=True)
    
    with vol.batch_upload(force=True) as batch:
        assert os.path.exists("modded-nanogpt.py"), "modded-nanogpt.py not found"
        batch.put_file("modded-nanogpt.py", "modded-nanogpt.py") 
        # remote path is /root/sandbox/modded-nanogpt.py when vol is mounted at /root/sandbox
    
    app = modal.App.lookup("my-app", create_if_missing=True)
    with modal.enable_output():
        
        image = Image.debian_slim(python_version="3.12")
        

        sb = Sandbox.create(
            app=app, 
            image=image,
            volumes={
                '/root/sandbox' : vol
            },
            timeout=30,
            verbose=True
        )

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    _root = "/root/sandbox"
    _run_dir = f"{_root}/runs/{ts}"
        
    
    # Prepare run directory and copy modded-nanogpt.py if present
    setup = (
        f"ls {_root} && "
        f"mkdir -p {_run_dir} && "
        f"cp {_root}/modded-nanogpt.py {_run_dir}/ && "
        f"cp {_root}/modded-nanogpt.py {_run_dir}/ "
    )
    
    command = 'ls -la'
    
    # Start a long-lived interactive bash in the run directory
    _process = sb.exec(
        "/bin/bash", "-c",
        f"{setup} && cd {_run_dir} && {command}",
        stderr=StreamType.STDOUT,
        stdout=StreamType.STDOUT
    )
    
    _process.wait()

    sb.terminate()
    
    sb.reload_volumes()

    

if __name__ == "__main__":
    test_sandbox()