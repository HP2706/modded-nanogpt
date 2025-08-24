import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask
from models.components.shared import CausalSelfAttention
from models.Nsa import NSA_Attention, create_sliding_mask

def create_dummy_block_mask(seq_len):
    """Create a simple causal block mask for testing."""
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    return create_block_mask(causal_mask, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)

def benchmark_attention(attn_module, x, ve, block_mask, warmup=10, repeats=50):
    """Benchmark a single attention module with warmup and multiple repeats."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = attn_module(x, ve, block_mask)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            _ = attn_module(x, ve, block_mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repeats

def run_benchmark(seq_lengths, hidden_dims, num_heads=12):
    """Run benchmarks for both attention mechanisms across different configurations."""
    results = {
        'causal': {},
        'nsa': {}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on {device}")
    
    for hidden_dim in hidden_dims:
        results['causal'][hidden_dim] = []
        results['nsa'][hidden_dim] = []
        
        for seq_len in seq_lengths:
            print(f"Benchmarking with seq_len={seq_len}, hidden_dim={hidden_dim}")
            
            # Create dummy inputs
            x = torch.randn(1, seq_len, hidden_dim, device=device)
            ve = torch.randn(1, seq_len, hidden_dim, device=device)
            
            sliding_window_mask = create_sliding_mask(seq_len, 32)
            
            block_mask = create_dummy_block_mask(seq_len)
            
            # Initialize models
            causal_attn = CausalSelfAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                layer_idx=0,
                head_dim=hidden_dim // num_heads
            ).to(device)
            
            nsa_attn = NSA_Attention(
                dim=hidden_dim,
                dim_head=hidden_dim // num_heads,
                heads=num_heads,
                layer_idx=0,
                sliding_window_size=32,
                compress_block_size=16,
                selection_block_size=16,
                num_selected_blocks=16,
                num_compressed_mem_kv=1,
                use_triton_kernel=True
            ).to(device)
            
            nsa_attn.forward
            
            # Benchmark
            causal_time = benchmark_attention(causal_attn, x, ve, block_mask)
            nsa_time = benchmark_attention(nsa_attn, x, ve, sliding_window_mask)
            
            results['causal'][hidden_dim].append(causal_time)
            results['nsa'][hidden_dim].append(nsa_time)
            
            print(f"  CausalSelfAttention: {causal_time*1000:.2f} ms")
            print(f"  NSA_Attention: {nsa_time*1000:.2f} ms")
            print(f"  Speedup: {causal_time/nsa_time:.2f}x" if nsa_time < causal_time else f"  Slowdown: {nsa_time/causal_time:.2f}x")
            
            # Free memory
            del causal_attn, nsa_attn
            torch.cuda.empty_cache()
    
    return results, seq_lengths

def plot_results(results, seq_lengths, hidden_dims):
    """Plot the benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Convert sequence lengths to readable format for plotting
    seq_labels = [f"{sl}" for sl in seq_lengths]
    
    # Plot time comparison for each hidden dimension
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 2, 2*i+1)
        plt.plot(seq_labels, [t*1000 for t in results['causal'][hidden_dim]], 'o-', label='CausalSelfAttention')
        plt.plot(seq_labels, [t*1000 for t in results['nsa'][hidden_dim]], 's-', label='NSA_Attention')
        plt.title(f'Latency (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot speedup/slowdown
        plt.subplot(len(hidden_dims), 2, 2*i+2)
        speedups = [c/n if n > 0 else 0 for c, n in zip(results['causal'][hidden_dim], results['nsa'][hidden_dim])]
        plt.bar(seq_labels, speedups)
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.title(f'NSA Speedup (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup (>1 means NSA is faster)')
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('attention_benchmark_results.png')
    plt.show()

def plot_memory_usage(seq_lengths, hidden_dims, num_heads=12):
    """Estimate and plot memory usage for both attention mechanisms."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_usage = {
        'causal': {},
        'nsa': {}
    }
    
    for hidden_dim in hidden_dims:
        memory_usage['causal'][hidden_dim] = []
        memory_usage['nsa'][hidden_dim] = []
        
        for seq_len in seq_lengths:
            # Create dummy inputs
            x = torch.randn(1, seq_len, hidden_dim, device=device)
            ve = torch.randn(1, seq_len, hidden_dim, device=device)
            block_mask = create_dummy_block_mask(seq_len)
            
            # Initialize models
            causal_attn = CausalSelfAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                layer_idx=0,
                head_dim=hidden_dim // num_heads
            ).to(device)
            
            nsa_attn = NSA_Attention(
                dim=hidden_dim,
                dim_head=hidden_dim // num_heads,
                heads=num_heads,
                layer_idx=0,
                sliding_window_size=32,
                compress_block_size=4,
                selection_block_size=4,
                num_selected_blocks=4,
                num_compressed_mem_kv=1
            ).to(device)
            
            # Measure memory before
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run causal attention
            with torch.no_grad():
                _ = causal_attn(x, ve, block_mask)
            
            causal_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            # Measure memory for NSA
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run NSA attention
            with torch.no_grad():
                _ = nsa_attn(x, ve, block_mask)
            
            nsa_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            memory_usage['causal'][hidden_dim].append(causal_mem)
            memory_usage['nsa'][hidden_dim].append(nsa_mem)
            
            # Free memory
            del causal_attn, nsa_attn
            torch.cuda.empty_cache()
    
    # Plot memory usage
    plt.figure(figsize=(15, 10))
    
    # Convert sequence lengths to readable format for plotting
    seq_labels = [f"{sl}" for sl in seq_lengths]
    
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 1, i+1)
        plt.plot(seq_labels, memory_usage['causal'][hidden_dim], 'o-', label='CausalSelfAttention')
        plt.plot(seq_labels, memory_usage['nsa'][hidden_dim], 's-', label='NSA_Attention')
        plt.title(f'Memory Usage (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('attention_memory_usage.png')
    plt.show()
    
    return memory_usage

if __name__ == "__main__":
    # Define sequence lengths and hidden dimensions to benchmark
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]  # 2^10 to 2^14
    hidden_dims = [768]  # Common model dimensions
    
    # Run benchmarks
    results, seq_lengths = run_benchmark(seq_lengths, hidden_dims, num_heads=8)
    
    # Plot results
    plot_results(results, seq_lengths, hidden_dims)
    
    # Plot memory usage
    memory_usage = plot_memory_usage(seq_lengths, hidden_dims)
    
    # Save results to file
    with open('benchmark_results.txt', 'w') as f:
        f.write("Benchmark Results\n")
        f.write("=================\n\n")
        
        for hidden_dim in hidden_dims:
            f.write(f"Hidden Dimension: {hidden_dim}\n")
            f.write("Sequence Length | CausalSelfAttention (ms) | NSA_Attention (ms) | Speedup | Memory Causal (MB) | Memory NSA (MB)\n")
            f.write("--------------- | ----------------------- | ----------------- | ------- | ------------------ | --------------\n")
            
            for i, seq_len in enumerate(seq_lengths):
                causal_time = results['causal'][hidden_dim][i] * 1000  # Convert to ms
                nsa_time = results['nsa'][hidden_dim][i] * 1000  # Convert to ms
                speedup = results['causal'][hidden_dim][i] / results['nsa'][hidden_dim][i] if nsa_time > 0 else 0
                causal_mem = memory_usage['causal'][hidden_dim][i]
                nsa_mem = memory_usage['nsa'][hidden_dim][i]
                
                f.write(f"{seq_len:15d} | {causal_time:23.2f} | {nsa_time:17.2f} | {speedup:7.2f} | {causal_mem:18.2f} | {nsa_mem:14.2f}\n")
            
            f.write("\n")


