import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask
from models.shared import CausalSelfAttention
from torch.nn.attention.flex_attention import flex_attention
from models.Nsa import NSA_Attention

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


def create_sliding_window_mask(seq_len, window_size, device=None):
    """
    Creates a causal sliding window attention mask.
    
    Args:
        seq_len (int): Length of the sequence
        window_size (int): Size of the sliding window
        device (torch.device, optional): Device to create the mask on
    
    Returns:
        torch.Tensor: Boolean mask of shape [seq_len, seq_len] where True values indicate
                     allowed attention connections
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Causal mask: can only attend to past tokens
    causal_mask = row_idx >= col_idx
    
    # Window mask: can only attend to tokens within the window
    window_mask = (row_idx - col_idx) <= window_size
    
    # Combined mask: both causal and within window
    mask = causal_mask & window_mask
    
    return mask

def create_dilated_sliding_window_mask(seq_len, window_size, dilation_rate=1, device=None):
    """
    Creates a dilated sliding window attention mask where tokens can attend to past tokens
    with regular gaps (dilation).
    
    Args:
        seq_len (int): Length of the sequence
        window_size (int): Size of the sliding window (in terms of attended positions)
        dilation_rate (int): Spacing between attended positions
        device (torch.device, optional): Device to create the mask on
    
    Returns:
        torch.Tensor: Boolean mask of shape [seq_len, seq_len] where True values indicate
                     allowed attention connections
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Causal mask: can only attend to past tokens
    causal_mask = row_idx >= col_idx
    
    # Calculate effective distance with dilation
    distance = (row_idx - col_idx) / dilation_rate
    
    # Window mask with dilation: can attend to tokens within dilated window
    # and only to positions that align with the dilation rate
    window_mask = (distance <= window_size) & ((row_idx - col_idx) % dilation_rate == 0)
    
    # Add the immediate context (previous token) regardless of dilation
    immediate_context = (row_idx - col_idx) == 1
    
    # Combined mask: causal and (within dilated window or immediate context)
    mask = causal_mask & (window_mask | immediate_context)
    
    return mask


class NSA_SlidingWindow(NSA_Attention):
    """NSA attention using only sliding window mechanism."""
    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # QKV projection
        q, k, v = torch.nn.functional.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = torch.nn.functional.rms_norm(q, (q.size(-1),)), torch.nn.functional.rms_norm(k, (k.size(-1),))
        
        # Rearrange dimensions
        q = self.flip_h_seq_dim(q)
        k = self.flip_h_seq_dim(k)
        v = self.flip_h_seq_dim(v)
        
        # Apply rotary embeddings
        q, k = self.rotary(q), self.rotary(k)
        
        # Handle value embeddings
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        
        # Only sliding window attention

        
        sliding_window_attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, scale=self.attn_scale).contiguous()
        
        # Return raw attention output without merge_heads and combine_heads
        return sliding_window_attn_out

class NSA_FineAttention(NSA_Attention):
    """NSA attention using only fine attention mechanism."""
    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1)
        device = x.device
        
        # Setup dimensions for fine attention
        compress_divisible_seq_len = (T // self.compress_block_size) * self.compress_block_size
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size
        fine_divisible_seq_len = ((T + self.selection_block_size - 1) // self.selection_block_size) * self.selection_block_size
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size
        
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # QKV projection
        q, k, v = torch.nn.functional.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = torch.nn.functional.rms_norm(q, (q.size(-1),)), torch.nn.functional.rms_norm(k, (k.size(-1),))
        
        # Rearrange dimensions
        q = self.flip_h_seq_dim(q)
        k = self.flip_h_seq_dim(k)
        v = self.flip_h_seq_dim(v)
        
        # Apply rotary embeddings
        q, k = self.rotary(q), self.rotary(k)
        
        # Handle value embeddings
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        
        # For fine attention, we need to create dummy importance scores
        # since we don't have compressed attention to guide selection
        # We'll select blocks in a regular pattern (e.g., every nth block)
        
        # Create dummy csim for fine attention
        csim = torch.randn(B, self.num_heads, T, num_compress_blocks + 1, device=device)
        
        # Only fine attention
        fine_attn_out = self.fine_attention(
            fq=q, 
            fk=k, 
            fv=v, 
            csim=csim, 
            num_mem_compress_kv=1, 
            num_compress_blocks=num_compress_blocks, 
            num_fine_blocks=num_fine_blocks, 
            fine_divisible_seq_len=fine_divisible_seq_len,
            device=device
        )
        
        # Return raw attention output without merge_heads and combine_heads
        return fine_attn_out

class NSA_CompressedAttention(NSA_Attention):
    """NSA attention using only compressed attention mechanism."""
    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1)
        device = x.device
        
        # Setup dimensions for compressed attention
        compress_divisible_seq_len = (T // self.compress_block_size) * self.compress_block_size
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size
        
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # QKV projection
        q, k, v = torch.nn.functional.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = torch.nn.functional.rms_norm(q, (q.size(-1),)), torch.nn.functional.rms_norm(k, (k.size(-1),))
        
        # Rearrange dimensions
        q = self.flip_h_seq_dim(q)
        k = self.flip_h_seq_dim(k)
        v = self.flip_h_seq_dim(v)
        
        # Handle value embeddings
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        
        # Memory tokens for compressed attention
        import einops
        mem_ck, mem_cv = einops.repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = B)
        num_mem_compress_kv = mem_ck.shape[-2]
        
        # Only compressed attention
        compressed_attn_out, _ = self.compressed_attention(
            q, 
            k, 
            v, 
            mem_ck, 
            mem_cv, 
            num_mem_compress_kv, 
            num_compress_blocks, 
            compress_divisible_seq_len, 
            device=device
        )
        
        # Return raw attention output without merge_heads and combine_heads
        return compressed_attn_out

def measure_memory_usage(attn_module, x, ve, block_mask):
    """Measure peak memory usage for an attention module."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = attn_module(x, ve, block_mask)
    
    return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

def run_benchmark(seq_lengths, hidden_dims, num_heads=8):
    """Run benchmarks for all attention mechanisms across different configurations."""
    results = {
        'causal': {},
        'sliding': {},
        'fine': {},
        'compressed': {},
        'nsa_full': {}
    }
    
    memory_usage = {
        'causal': {},
        'sliding': {},
        'fine': {},
        'compressed': {},
        'nsa_full': {}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on {device}")
    
    for hidden_dim in hidden_dims:
        results['causal'][hidden_dim] = []
        results['sliding'][hidden_dim] = []
        results['fine'][hidden_dim] = []
        results['compressed'][hidden_dim] = []
        results['nsa_full'][hidden_dim] = []
        
        memory_usage['causal'][hidden_dim] = []
        memory_usage['sliding'][hidden_dim] = []
        memory_usage['fine'][hidden_dim] = []
        memory_usage['compressed'][hidden_dim] = []
        memory_usage['nsa_full'][hidden_dim] = []
        
        for seq_len in seq_lengths:
            print(f"Benchmarking with seq_len={seq_len}, hidden_dim={hidden_dim}")
            block_mask = create_dummy_block_mask(seq_len)
            sliding_window_mask = create_sliding_window_mask(seq_len, 32)
            
            # Create dummy inputs
            x = torch.randn(1, seq_len, hidden_dim, device=device)
            ve = torch.randn(1, seq_len, hidden_dim, device=device)
            
            # Initialize models
            causal_attn = CausalSelfAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                layer_idx=0,
                head_dim=hidden_dim // num_heads
            ).to(device)
            
            sliding_attn = NSA_SlidingWindow(
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
            
            fine_attn = NSA_FineAttention(
                dim=hidden_dim,
                dim_head=hidden_dim // num_heads,
                heads=num_heads,
                layer_idx=0,
                sliding_window_size=128,
                compress_block_size=4,
                selection_block_size=4,
                num_selected_blocks=4,
                num_compressed_mem_kv=1
            ).to(device)
            
            compressed_attn = NSA_CompressedAttention(
                dim=hidden_dim,
                dim_head=hidden_dim // num_heads,
                heads=num_heads,
                layer_idx=0,
                sliding_window_size=128,
                compress_block_size=4,
                selection_block_size=4,
                num_selected_blocks=4,
                num_compressed_mem_kv=1
            ).to(device)
            
            
            # Benchmark latency
            causal_time = benchmark_attention(causal_attn, x, ve, block_mask)
            sliding_time = benchmark_attention(sliding_attn, x, ve, sliding_window_mask)
            fine_time = benchmark_attention(fine_attn, x, ve, block_mask)
            compressed_time = benchmark_attention(compressed_attn, x, ve, block_mask)
            
            results['causal'][hidden_dim].append(causal_time)
            results['sliding'][hidden_dim].append(sliding_time)
            results['fine'][hidden_dim].append(fine_time)
            results['compressed'][hidden_dim].append(compressed_time)
            
            # Measure memory usage
            causal_mem = measure_memory_usage(causal_attn, x, ve, block_mask)
            sliding_mem = measure_memory_usage(sliding_attn, x, ve, sliding_window_mask)
            fine_mem = measure_memory_usage(fine_attn, x, ve, block_mask)
            compressed_mem = measure_memory_usage(compressed_attn, x, ve, block_mask)
            
            memory_usage['causal'][hidden_dim].append(causal_mem)
            memory_usage['sliding'][hidden_dim].append(sliding_mem)
            memory_usage['fine'][hidden_dim].append(fine_mem)
            memory_usage['compressed'][hidden_dim].append(compressed_mem)
            
            print(f"  CausalSelfAttention: {causal_time*1000:.2f} ms, {causal_mem:.2f} MB")
            print(f"  NSA Sliding Window: {sliding_time*1000:.2f} ms, {sliding_mem:.2f} MB")
            print(f"  NSA Fine Attention: {fine_time*1000:.2f} ms, {fine_mem:.2f} MB")
            print(f"  NSA Compressed Attention: {compressed_time*1000:.2f} ms, {compressed_mem:.2f} MB")
            
            # Free memory
            del causal_attn, sliding_attn, fine_attn, compressed_attn
            torch.cuda.empty_cache()
    
    return results, memory_usage, seq_lengths

def plot_results(results, memory_usage, seq_lengths, hidden_dims):
    """Plot the benchmark results."""
    # Convert sequence lengths to readable format for plotting
    seq_labels = [f"{sl}" for sl in seq_lengths]
    
    # Plot latency comparison
    plt.figure(figsize=(15, 10))
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 1, i+1)
        plt.plot(seq_labels, [t*1000 for t in results['causal'][hidden_dim]], 'o-', label='CausalSelfAttention')
        plt.plot(seq_labels, [t*1000 for t in results['sliding'][hidden_dim]], 's-', label='NSA Sliding Window')
        plt.plot(seq_labels, [t*1000 for t in results['fine'][hidden_dim]], '^-', label='NSA Fine Attention')
        plt.plot(seq_labels, [t*1000 for t in results['compressed'][hidden_dim]], 'D-', label='NSA Compressed Attention')
        plt.title(f'Latency Comparison (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('nsa_mechanisms_latency.png')
    
    # Plot memory usage comparison
    plt.figure(figsize=(15, 10))
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 1, i+1)
        plt.plot(seq_labels, memory_usage['causal'][hidden_dim], 'o-', label='CausalSelfAttention')
        plt.plot(seq_labels, memory_usage['sliding'][hidden_dim], 's-', label='NSA Sliding Window')
        plt.plot(seq_labels, memory_usage['fine'][hidden_dim], '^-', label='NSA Fine Attention')
        plt.plot(seq_labels, memory_usage['compressed'][hidden_dim], 'D-', label='NSA Compressed Attention')
        plt.plot(seq_labels, memory_usage['nsa_full'][hidden_dim], '*-', label='NSA Full (all mechanisms)')
        plt.title(f'Memory Usage Comparison (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('nsa_mechanisms_memory.png')
    
    # Plot relative speedup compared to causal attention
    plt.figure(figsize=(15, 15))
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 1, i+1)
        
        # Calculate speedups relative to causal attention
        sliding_speedup = [c/s if s > 0 else 0 for c, s in zip(results['causal'][hidden_dim], results['sliding'][hidden_dim])]
        fine_speedup = [c/f if f > 0 else 0 for c, f in zip(results['causal'][hidden_dim], results['fine'][hidden_dim])]
        compressed_speedup = [c/co if co > 0 else 0 for c, co in zip(results['causal'][hidden_dim], results['compressed'][hidden_dim])]
        nsa_full_speedup = [c/n if n > 0 else 0 for c, n in zip(results['causal'][hidden_dim], results['nsa_full'][hidden_dim])]
        
        # Create grouped bar chart
        x = np.arange(len(seq_lengths))
        width = 0.2
        
        plt.bar(x - 1.5*width, sliding_speedup, width, label='NSA Sliding Window')
        plt.bar(x - 0.5*width, fine_speedup, width, label='NSA Fine Attention')
        plt.bar(x + 0.5*width, compressed_speedup, width, label='NSA Compressed Attention')
        plt.bar(x + 1.5*width, nsa_full_speedup, width, label='NSA Full (all mechanisms)')
        
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.title(f'Speedup vs CausalSelfAttention (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup (>1 means faster than causal)')
        plt.xticks(x, seq_labels, rotation=45)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nsa_mechanisms_speedup.png')
    
    # Plot memory efficiency compared to causal attention
    plt.figure(figsize=(15, 15))
    for i, hidden_dim in enumerate(hidden_dims):
        plt.subplot(len(hidden_dims), 1, i+1)
        
        # Calculate memory efficiency relative to causal attention
        sliding_efficiency = [c/s if s > 0 else 0 for c, s in zip(memory_usage['causal'][hidden_dim], memory_usage['sliding'][hidden_dim])]
        fine_efficiency = [c/f if f > 0 else 0 for c, f in zip(memory_usage['causal'][hidden_dim], memory_usage['fine'][hidden_dim])]
        compressed_efficiency = [c/co if co > 0 else 0 for c, co in zip(memory_usage['causal'][hidden_dim], memory_usage['compressed'][hidden_dim])]
        nsa_full_efficiency = [c/n if n > 0 else 0 for c, n in zip(memory_usage['causal'][hidden_dim], memory_usage['nsa_full'][hidden_dim])]
        
        # Create grouped bar chart
        x = np.arange(len(seq_lengths))
        width = 0.2
        
        plt.bar(x - 1.5*width, sliding_efficiency, width, label='NSA Sliding Window')
        plt.bar(x - 0.5*width, fine_efficiency, width, label='NSA Fine Attention')
        plt.bar(x + 0.5*width, compressed_efficiency, width, label='NSA Compressed Attention')
        plt.bar(x + 1.5*width, nsa_full_efficiency, width, label='NSA Full (all mechanisms)')
        
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.title(f'Memory Efficiency vs CausalSelfAttention (dim={hidden_dim})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Efficiency (>1 means more efficient than causal)')
        plt.xticks(x, seq_labels, rotation=45)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nsa_mechanisms_memory_efficiency.png')

def save_results_to_file(results, memory_usage, seq_lengths, hidden_dims):
    """Save benchmark results to a text file."""
    with open('nsa_mechanisms_benchmark_results.txt', 'w') as f:
        f.write("NSA Mechanisms Benchmark Results\n")
        f.write("===============================\n\n")
        f.write("Note: All measurements exclude merge_heads and combine_heads operations\n\n")
        
        for hidden_dim in hidden_dims:
            f.write(f"Hidden Dimension: {hidden_dim}\n")
            f.write("Sequence Length | Causal (ms) | Sliding (ms) | Fine (ms) | Compressed (ms) | NSA Full (ms)\n")
            f.write("--------------- | ----------- | ------------ | --------- | --------------- | ------------\n")
            
            for i, seq_len in enumerate(seq_lengths):
                causal_time = results['causal'][hidden_dim][i] * 1000
                sliding_time = results['sliding'][hidden_dim][i] * 1000
                fine_time = results['fine'][hidden_dim][i] * 1000
                compressed_time = results['compressed'][hidden_dim][i] * 1000
                nsa_full_time = results['nsa_full'][hidden_dim][i] * 1000
                
                f.write(f"{seq_len:15d} | {causal_time:11.2f} | {sliding_time:12.2f} | {fine_time:9.2f} | {compressed_time:15.2f} | {nsa_full_time:12.2f}\n")
            
            f.write("\n")
            
            f.write("Memory Usage (MB)\n")
            f.write("Sequence Length | Causal | Sliding | Fine | Compressed | NSA Full\n")
            f.write("--------------- | ------ | ------- | ---- | ---------- | --------\n")
            
            for i, seq_len in enumerate(seq_lengths):
                causal_mem = memory_usage['causal'][hidden_dim][i]
                sliding_mem = memory_usage
                
                
if __name__ == "__main__":
    seq_lengths = [1024, 2048, 4096, 8192, 16384]
    hidden_dims = [768]
    num_heads = 12
    
    results, memory_usage, seq_lengths = run_benchmark(seq_lengths, hidden_dims, num_heads)
    plot_results(results, memory_usage, seq_lengths, hidden_dims)