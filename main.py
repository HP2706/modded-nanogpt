from modal_common import volume, image, app, ssh_function_wrapper
from modal import Secret, gpu

@app.function(
    gpu=gpu.H100(),
    image=image, 
    timeout=60*60, # 1 hour
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def train_gpt():
    import subprocess
    # run the test script
    subprocess.run([
        "torchrun", "/root/project/train_gpt.py"
    ], check=True)
    

KILL_AFTER = 60 * 60 * 14 # 14 hours
@app.function(
    gpu=gpu.A100(),
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def ssh_function():
    ssh_function_wrapper()