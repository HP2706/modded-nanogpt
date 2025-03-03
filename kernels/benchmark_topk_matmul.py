import torch
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from typing import List, Tuple, Dict
from topk_matmul import topk_matmul, torch_topk_matmul
from dataclasses import dataclass

@dataclass
class MatrixDims:
    M: int  # Rows of A
    K: int  # Cols of A / Rows of B
    N: int  # Cols of B
    
def benchmark_size_scaling(
    base_sizes: List[int], 
    aspect_ratios: List[Tuple[float, float, float]] = [(1.0, 1.0, 1.0)],
    k: int = 16, 
    num_trials: int = 10,
    device: str = "cuda"
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Benchmark how performance scales with matrix size for different aspect ratios.
    
    Args:
        base_sizes: Base sizes to scale matrices
        aspect_ratios: List of (M_ratio, K_ratio, N_ratio) tuples
        k: Number of top values to keep
        num_trials: Number of trials for each configuration
        device: Device to run on
    """
    results = {}
    
    for M_ratio, K_ratio, N_ratio in aspect_ratios:
        ratio_name = f"M{M_ratio:.1f}_K{K_ratio:.1f}_N{N_ratio:.1f}"
        print(f"\nBenchmarking aspect ratio: {ratio_name}")
        
        triton_times = []
        torch_times = []
        
        for base_size in base_sizes:
            dims = MatrixDims(
                M=int(base_size * M_ratio),
                K=int(base_size * K_ratio),
                N=int(base_size * N_ratio)
            )
            print(f"  Size (M={dims.M}, K={dims.K}, N={dims.N})...")
            
            triton_trial_times = []
            torch_trial_times = []
            
            # Generate random matrices
            a = torch.randn((dims.M, dims.K), dtype=torch.float32, device=device)
            b = torch.randn((dims.N, dims.K), dtype=torch.float32, device=device)
            
            # Warmup
            _ = topk_matmul(a, b.T, k)
            _ = torch_topk_matmul(a, b.T, k)
            torch.cuda.synchronize()
            
            for _ in range(num_trials):
                # Benchmark Triton implementation
                start = perf_counter()
                _ = topk_matmul(a, b.T, k)
                torch.cuda.synchronize()
                triton_trial_times.append(perf_counter() - start)
                
                # Benchmark PyTorch implementation
                start = perf_counter()
                _ = torch_topk_matmul(a, b.T, k)
                torch.cuda.synchronize()
                torch_trial_times.append(perf_counter() - start)
            
            # Record median times
            triton_times.append(np.median(triton_trial_times))
            torch_times.append(np.median(torch_trial_times))
        
        triton_times = np.array(triton_times)
        torch_times = np.array(torch_times)
        speedups = torch_times / triton_times
        
        results[ratio_name] = (triton_times, torch_times, speedups)
    
    return results

def benchmark_k_scaling(
    dims: MatrixDims,
    k_values: List[int] = None,
    num_trials: int = 10,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark how performance scales with k for fixed matrix dimensions."""
    if k_values is None:
        k_values = [2**i for i in range(1, min(11, int(np.log2(dims.N)) + 1))]
    
    triton_times = []
    torch_times = []
    
    # Generate random matrices
    a = torch.randn((dims.M, dims.K), dtype=torch.float32, device=device)
    b = torch.randn((dims.N, dims.K), dtype=torch.float32, device=device)
    
    for k in k_values:
        print(f"Benchmarking k={k}...")
        triton_trial_times = []
        torch_trial_times = []
        
        # Warmup
        _ = topk_matmul(a, b.T, k)
        _ = torch_topk_matmul(a, b.T, k)
        torch.cuda.synchronize()
        
        for _ in range(num_trials):
            # Benchmark Triton implementation
            start = perf_counter()
            _ = topk_matmul(a, b.T, k)
            torch.cuda.synchronize()
            triton_trial_times.append(perf_counter() - start)
            
            # Benchmark PyTorch implementation
            start = perf_counter()
            _ = torch_topk_matmul(a, b.T, k)
            torch.cuda.synchronize()
            torch_trial_times.append(perf_counter() - start)
        
        # Record median times
        triton_times.append(np.median(triton_trial_times))
        torch_times.append(np.median(torch_trial_times))
    
    triton_times = np.array(triton_times)
    torch_times = np.array(torch_times)
    speedups = torch_times / triton_times
    
    return triton_times, torch_times, speedups

def plot_size_scaling_results(
    base_sizes: List[int],
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    k: int,
    save_path: str = None
):
    """Create subplots for different aspect ratios."""
    n_ratios = len(results)
    fig, axes = plt.subplots(n_ratios, 1, figsize=(12, 6*n_ratios))
    if n_ratios == 1:
        axes = [axes]
    
    for (ratio_name, (triton_times, torch_times, speedups)), ax in zip(results.items(), axes):
        # Plot timings
        ax1 = ax
        l1 = ax1.plot(base_sizes, triton_times * 1000, 'b-', label='Triton')
        l2 = ax1.plot(base_sizes, torch_times * 1000, 'r-', label='PyTorch')
        ax1.set_xlabel('Base Matrix Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot speedup on secondary axis
        ax2 = ax1.twinx()
        l3 = ax2.plot(base_sizes, speedups, 'g--', label='Speedup')
        ax2.set_ylabel('Speedup (PyTorch/Triton)')
        
        # Combine legends
        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
        
        ax1.set_title(f'Performance Scaling - {ratio_name} (k={k})')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_k_scaling_results(
    k_values: List[int],
    triton_times: np.ndarray,
    torch_times: np.ndarray,
    speedups: np.ndarray,
    dims: MatrixDims,
    save_path: str = None
):
    """Create a dual-axis plot showing timing and speedup for k scaling."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot timings
    l1 = ax1.plot(k_values, triton_times * 1000, 'b-', label='Triton')
    l2 = ax1.plot(k_values, torch_times * 1000, 'r-', label='PyTorch')
    ax1.set_xlabel('K Value')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Plot speedup on secondary axis
    ax2 = ax1.twinx()
    l3 = ax2.plot(k_values, speedups, 'g--', label='Speedup')
    ax2.set_ylabel('Speedup (PyTorch/Triton)')
    
    # Combine legends
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    plt.title(f'TopK MatMul Performance vs K (M={dims.M}, K={dims.K}, N={dims.N})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = "cpu"
    else:
        device = "cuda"
    
    # Test different aspect ratios
    base_sizes = [128, 256, 512, 1024, 2048]
    aspect_ratios = [
        (1.0, 1.0, 1.0),    # Square matrices
        (0.5, 1.0, 2.0),    # Wide output
        (2.0, 1.0, 0.5),    # Tall output
        (1.0, 0.5, 1.0),    # Thin inner dimension
    ]
    k = 16
    
    # Benchmark size scaling
    results = benchmark_size_scaling(base_sizes, aspect_ratios, k, device=device)
    plot_size_scaling_results(base_sizes, results, k, "size_scaling_benchmark.png")
    
    # Benchmark k scaling for a specific matrix size and aspect ratio
    dims = MatrixDims(M=1024, K=512, N=2048)  # Example dimensions
    k_values = [2**i for i in range(1, min(11, int(np.log2(dims.N)) + 1))]
    triton_times, torch_times, speedups = benchmark_k_scaling(dims, k_values, device=device)
    plot_k_scaling_results(
        k_values,
        triton_times,
        torch_times,
        speedups,
        dims,
        "k_scaling_benchmark.png"
    ) 