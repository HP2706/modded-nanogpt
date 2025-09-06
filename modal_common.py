# use modal ssh
import modal
from modal.volume import FileEntry, FileEntryType
from modal import Image, gpu, Secret, Volume, App
import sys
import os
import inspect
from modal_ssh import ssh_function_wrapper, maybe_upload_project, configure_ssh_image




app = App(name="gpt2-speedrun")
volume = Volume.from_name("gpt2-speedrun", create_if_missing=True)


cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

#get to root of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
    ).add_local_file(
        'requirements.txt',
        '/root/requirements.txt',
        copy=True
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
    ).uv_pip_install(
        requirements=['requirements.txt']
    ).env(
        {
            "HUGGINGFACE_HUB_CACHE": "/root/models/hf"
        }
    ).run_commands(
    "pip install git+https://github.com/HP2706/modal-ssh.git"
)
)

image = configure_ssh_image(image)



KILL_AFTER = 60 * 60 * 14 # 14 hours
@app.function(
    gpu='T4',
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def ssh_function():
    ssh_function_wrapper()
    
@app.function(
    gpu='H100',
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def train_gpt(**kwargs):
    """
    This function accepts the same arguments as the train function.
    """
    import subprocess
    import os
    
    # Change to project directory
    os.chdir("/root/data/data/project")
    
    kwargs_str = ' '.join([f'--{k} {v}' for k, v in kwargs.items()])
    print(f"Running train with: {kwargs_str}")
    os.system(f"torchrun train_gpt.py {kwargs_str}")


