from modal_common import volume, image, app, ssh_function_wrapper, maybe_upload_project, train_gpt
from modal import Secret
import argparse
import argparse
import torch

from modal import Volume

image = image.add_local_dir(
    '.',
    '/root',
    ignore=['.git', '.venv', '.DS_Store', '__pycache__'],
    copy=True
)

#data/project/data/fineweb10B
fineweb10B_volume = Volume.from_name("fineweb10B", create_if_missing=True)

KILL_AFTER = 60 * 60 * 14 # 14 hours
@app.function(
    gpu='A10G',
    image=image, 
    timeout=KILL_AFTER,
    secrets=[Secret.from_name("wandb"), Secret.from_name("HF_SECRET")],
    volumes={
        '/root/data': volume,
    }
)
def ssh_function():
    ssh_function_wrapper()
    
@app.function(
    volumes={
        '/root/fineweb10B': fineweb10B_volume,
        '/root/data': volume
    }
)
def cp_data_to_fineweb10B():
    print("os.listdir('/root'):", os.listdir('/root'))
    os.chdir('/root/modded-nanogpt')
    import subprocess
    output = subprocess.run(["cp", "-r", "/root/data/data/project/data/fineweb10B", "/root/fineweb10B"], stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    print(output)
    import os
    print(os.listdir("/root/fineweb10B"))
    

@app.local_entrypoint()
def main():
    # Method 4: Modify GPU after function creation (most direct approach)
    """ train_gpt.remote(
        type='nsa',
        #torch_compile=True,
    ) """
    cp_data_to_fineweb10B.remote()
