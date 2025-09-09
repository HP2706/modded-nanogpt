import modal
from modal import Volume
from modal import Secret

agent_volume = Volume.from_name("mle-sandbox", create_if_missing=True)
fineweb10B_volume = Volume.from_name("fineweb10B", create_if_missing=True)    

# Use a fresh app object for local runs to avoid reusing a running app
modal_app = modal.App("speedrun-local")
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
).uv_pip_install(
    "torch",
    "transformers",
    "datasets",
    "einops",
    "psutil",
    "hf-transfer",
    "fire",
    "einx",
    "matplotlib",
    "torchvision",
    "wandb",
)

@modal_app.function(
    image=image,
    volumes={
        '/root/data': fineweb10B_volume,
        '/root/sandbox': agent_volume
    },
    gpu='A100-80GB:1',
    secrets=[Secret.from_name("wandb")]
)
def test_script():
    import subprocess
    cmd = "ls -la /root/sandbox"
    process = subprocess.Popen(cmd, shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd='/root',
        universal_newlines=True,
        bufsize=1
    )
    print(process.stdout.read())
    
    
    #cmd = "python /root/sandbox/cifar_speedrun_unoptimized.py"
    cmd = "torchrun /root/sandbox/modded_nanogpt_unoptimized.py"
    #cmd = "torchrun /root/sandbox/modded_nanogpt.py"

    process = subprocess.Popen(cmd, shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd='/root',
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Wait for process to complete and get return code
    process.wait()

@modal_app.local_entrypoint()
def main():
    # Launch the Modal function from a proper local entrypoint
    test_script.remote()
