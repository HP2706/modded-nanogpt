from modal_common import volume, image, app, ssh_function_wrapper, maybe_upload_project, train_gpt
from modal import Secret, gpu

KILL_AFTER = 60 * 60 * 14 # 14 hours
@app.function(
    gpu='A10G',
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume
    }
)
def ssh_function():
    ssh_function_wrapper()
    

@app.local_entrypoint()
def main():
    maybe_upload_project()
    ssh_function.remote()
    
    #train_gpt.remote(
    #    type='current-best',
    #    torch_compile=True,
    #)
