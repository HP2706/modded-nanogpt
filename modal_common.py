# use modal ssh
import modal
from modal import Image, gpu, Secret, Volume, App
import sys
import os

app = App(name="gpt2-speedrun")
volume = Volume.from_name("gpt2-speedrun", create_if_missing=True)


#get to root of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image = Image.debian_slim(python_version="3.12").apt_install(
    "openssh-server",
    "git"
    ).run_commands(
        "mkdir /run/sshd",
    ).copy_local_file(
        "~/.ssh/id_rsa.pub", "/root/.ssh/authorized_keys"
    ).pip_install(
        "uv",
    ).env(
        {
            "VIRTUAL_ENV": "/usr/local",
            "PATH" : "/usr/local/bin:$PATH",
            "HF_HUB_ENABLE_HF_TRANSFER" : "1",
            "GIT_USER_NAME": os.environ.get("GIT_USER_NAME", "Default Name"),
            "GIT_USER_EMAIL": os.environ.get("GIT_USER_EMAIL", "default@example.com")
        }
    ).run_commands(
        "git config --global user.name \"$GIT_USER_NAME\"",
        "git config --global user.email \"$GIT_USER_EMAIL\""
    ).copy_local_dir(
        '.',
        remote_path="/root/project"
    ).run_commands(
        "uv pip install --pre torch==2.7.0.dev20250110+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade",
        "uv pip install -r /root/project/requirements.txt"
    ).env(
        {
            "HUGGINGFACE_HUB_CACHE": "/root/models/hf"
        }
    )


def ssh_function_wrapper():
    import subprocess
    import time
    import modal
    import signal
    import torch
    import gc
    import os
    import atexit

    def cleanup_gpu(signum=None, frame=None):
        try:
            # Clear CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # More graceful process termination
            os.system('pkill -15 -f python')  # Send SIGTERM first
            time.sleep(2)  # Give processes time to cleanup
            current_pid = os.getpid()
            os.system(f'pkill -9 -f python && kill -9 $(pgrep -f python | grep -v {current_pid})')
            
            if torch.cuda.is_available():
                os.system('nvidia-smi --gpu-reset')
        except Exception as e:
            print(f"Cleanup error: {e}")

    # Register cleanup for various signals and exit
    signal.signal(signal.SIGHUP, cleanup_gpu)
    signal.signal(signal.SIGTERM, cleanup_gpu)
    signal.signal(signal.SIGINT, cleanup_gpu)
    atexit.register(cleanup_gpu)

    # Configure sshd with custom settings
    sshd_config = """
    PrintMotd no
    PrintLastLog no
    UsePAM no
    """
    with open("/etc/ssh/sshd_config.d/custom.conf", "w") as f:
        f.write(sshd_config)

    try:
        subprocess.run(["service", "ssh", "restart"], check=True)
        with modal.forward(port=22, unencrypted=True) as tunnel:
            hostname, port = tunnel.tcp_socket
            connection_cmd = f'ssh -p {port} root@{hostname}'
            print(f"ssh into container using: {connection_cmd}")
            
            while True:
                time.sleep(60)  # Check every minute
                # Verify SSH daemon is still running
                try:
                    subprocess.run(["pgrep", "sshd"], check=True)
                except subprocess.CalledProcessError:
                    print("SSH daemon died, restarting...")
                    subprocess.run(["service", "ssh", "restart"], check=True)
    except Exception as e:
        print(f"SSH server error: {e}")
    finally:
        cleanup_gpu()
        volume.commit()

KILL_AFTER = 60 * 60 * 14 # 14 hours
@app.function(
    gpu=gpu.A10G(),
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def ssh_function():
    ssh_function_wrapper()