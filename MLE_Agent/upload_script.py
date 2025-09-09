from modal import Volume

agent_volume = Volume.from_name("mle-sandbox", create_if_missing=True)
fineweb10B_volume = Volume.from_name("fineweb10B", create_if_missing=True)

# todo USE https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with agent_volume.batch_upload(force=True) as batch:
    batch.put_file("environments/modded_nanogpt.py", "modded_nanogpt.py") 
    batch.put_file("environments/modded_nanogpt_unoptimized.py", "modded_nanogpt_unoptimized.py")
    batch.put_file("environments/cifar_speedrun_unoptimized.py", "cifar_speedrun_unoptimized.py") 
    batch.put_file('environments/cifar_speedrun.py', 'cifar_speedrun.py')
    

