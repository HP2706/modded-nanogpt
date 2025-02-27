import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from models.ultra_sparse_mem import UltraSparseMemoryMLP
from models.memory_at_scale import MemoryAtScaleMLP
from models.peerlayer import PeerLayer
import gc
from torch.cuda import max_memory_allocated, reset_peak_memory_stats


def measure_time_and_memory(model, inputs, n_runs=10, warmup=3):
    """Measure inference time and memory usage of a model."""
    device = next(model.parameters()).device
    
    # Ensure inputs are on the same device as the model
    if isinstance(inputs, tuple):
        inputs = tuple(x.to(device) for x in inputs)
    else:
        inputs = inputs.to(device)
    
    # Warmup
    for _ in range(warmup):
        if isinstance(model, MemoryAtScaleMLP):
            indices, per_sample_weights = inputs
            _ = model(indices, per_sample_weights)
        elif isinstance(model, UltraSparseMemoryMLP):
            Q = inputs
            _ = model(Q)
        elif isinstance(model, PeerLayer):
            x = inputs
            _ = model(x)
    
    # Synchronize before measuring
    if device.type == 'cuda':
        torch.cuda.synchronize()
        reset_peak_memory_stats()
    
    # Measure time
    start_time = time.time()
    for _ in range(n_runs):
        if isinstance(model, MemoryAtScaleMLP):
            indices, per_sample_weights = inputs
            _ = model(indices, per_sample_weights)
        elif isinstance(model, UltraSparseMemoryMLP):
            Q = inputs
            _ = model(Q)
        elif isinstance(model, PeerLayer):
            x = inputs
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs
    
    # Measure memory
    if device.type == 'cuda':
        memory_usage = max_memory_allocated() / (1024 ** 2)  # Convert to MB
    else:
        memory_usage = 0  # Not measuring memory for CPU
    
    return avg_time, memory_usage


def run_benchmark(batch_size, seq_len, d_model, n_experts, device='cuda'):
    """Run benchmark for all three models with the same parameters."""
    print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, n_experts={n_experts}")
    
    # Common parameters
    d_query = d_model // 2
    query_heads = 4
    topk = 10
    
    # Model-specific parameters
    r = 4  # For UltraSparseMemoryMLP
    
    # Initialize models
    memory_at_scale = MemoryAtScaleMLP(
        d_model=d_model,
        n_experts=n_experts
    ).to(device)
    
    ultra_sparse = UltraSparseMemoryMLP(
        d_value=d_model,
        d_query=d_query,
        r=r,
        query_heads=query_heads,
        n_experts=n_experts,
        topk=topk
    ).to(device)
    
    peer_layer = PeerLayer(
        d_in=d_model,
        d_out=d_model,
        d_query=d_query,
        query_heads=query_heads,
        n_experts=n_experts,
        topk=topk
    ).to(device)
    
    
    print("number of parameters")
    print(f"MemoryAtScaleMLP: {sum(p.numel() for p in memory_at_scale.parameters())/1e6} M")
    print(f"UltraSparseMemoryMLP: {sum(p.numel() for p in ultra_sparse.parameters())/1e6} M")
    print(f"PeerLayer: {sum(p.numel() for p in peer_layer.parameters())/1e6} M")
    # Prepare inputs for each model
    # For MemoryAtScaleMLP
    indices = torch.randint(0, n_experts, (batch_size * seq_len, topk), device=device)
    per_sample_weights = torch.randn((batch_size * seq_len, topk), device=device, requires_grad=True)
    
    # For UltraSparseMemoryMLP
    Q = torch.randn((batch_size, seq_len, 1, d_query), device=device)  # Assuming expansion_factor=1
    
    # For PeerLayer
    x = torch.randn((batch_size, seq_len, d_model), device=device)
    
    # Measure performance
    results = {}
    
    # Memory at Scale
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        reset_peak_memory_stats()
    
    time_mas, memory_mas = measure_time_and_memory(
        memory_at_scale, 
        (indices, per_sample_weights)
    )
    results['Memory at Scale'] = {'time': time_mas, 'memory': memory_mas}
    
    # Ultra Sparse Memory
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        reset_peak_memory_stats()
    
    time_usm, memory_usm = measure_time_and_memory(
        ultra_sparse, 
        Q
    )
    results['Ultra Sparse Memory'] = {'time': time_usm, 'memory': memory_usm}
    
    # Peer Layer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        reset_peak_memory_stats()
    
    time_peer, memory_peer = measure_time_and_memory(
        peer_layer, 
        x
    )
    results['Peer Layer'] = {'time': time_peer, 'memory': memory_peer}
    
    # Print results
    print("\nResults:")
    print(f"{'Model':<20} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['time']*1000:<15.2f} {metrics['memory']:<15.2f}")
    
    return results


def plot_results(all_results, param_values, param_name):
    """Plot time and memory results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = list(all_results[0].keys())
    x = np.arange(len(param_values))
    width = 0.25
    
    # Time plot
    for i, model in enumerate(models):
        times = [result[model]['time'] * 1000 for result in all_results]  # Convert to ms
        ax1.bar(x + i*width - width, times, width, label=model)
    
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Inference Time vs {param_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_values)
    ax1.legend()
    
    # Memory plot
    for i, model in enumerate(models):
        memory = [result[model]['memory'] for result in all_results]
        ax2.bar(x + i*width - width, memory, width, label=model)
    
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title(f'Memory Usage vs {param_name}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_values)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'benchmark_{param_name}.png')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    
    # Default parameters
    default_batch_size = 32
    default_seq_len = 128
    default_d_model = 512
    default_n_experts = 256**2
    
    # Benchmark varying batch sizes
    batch_sizes = [8, 16, 32, 64]
    batch_results = []
    for bs in batch_sizes:
        result = run_benchmark(
            batch_size=bs,
            seq_len=default_seq_len,
            d_model=default_d_model,
            n_experts=default_n_experts,
            device=device
        )
        batch_results.append(result)
    
    plot_results(batch_results, batch_sizes, 'Batch Size')
    
    # Benchmark varying sequence lengths
    seq_lengths = [64, 128, 256, 512]
    seq_results = []
    for sl in seq_lengths:
        result = run_benchmark(
            batch_size=default_batch_size,
            seq_len=sl,
            d_model=default_d_model,
            n_experts=default_n_experts,
            device=device
        )
        seq_results.append(result)
    
    plot_results(seq_results, seq_lengths, 'Sequence Length')
    
    # Benchmark varying model dimensions
    d_models = [256, 512, 1024, 2048]
    dim_results = []
    for dm in d_models:
        result = run_benchmark(
            batch_size=default_batch_size,
            seq_len=default_seq_len,
            d_model=dm,
            n_experts=default_n_experts,
            device=device
        )
        dim_results.append(result)
    
    plot_results(dim_results, d_models, 'Model Dimension')
    
    # Benchmark varying number of experts
    n_experts_list = [64, 256, 1024, 4096]
    expert_results = []
    for ne in n_experts_list:
        result = run_benchmark(
            batch_size=default_batch_size,
            seq_len=default_seq_len,
            d_model=default_d_model,
            n_experts=ne,
            device=device
        )
        expert_results.append(result)
    
    plot_results(expert_results, n_experts_list, 'Number of Experts')


if __name__ == "__main__":
    main()

