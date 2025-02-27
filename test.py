import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Nsa import NSA_Attention
from einops import repeat, rearrange
from torch.nn.attention.flex_attention import create_block_mask

# Setup a small test for NSA compressed attention and fine attention

def test_nsa_components():
    # Parameters
    batch_size = 1
    seq_len = 1024
    model_dim = 768
    num_heads = 12
    head_dim = model_dim // num_heads
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Create NSA_Attention module
    attn = NSA_Attention(
        dim=model_dim,
        dim_head=head_dim,
        heads=num_heads,
        layer_idx=0,
        sliding_window_size=32,
        compress_block_size=4,
        selection_block_size=4,
        num_selected_blocks=4,
        query_heads_share_selected_kv=True
    ).to(device)
    
    # Create random input tensors
    x = torch.randn(batch_size, seq_len, model_dim).to(device)
    
    # Process input to get q, k, v
    B, T = x.size(0), x.size(1)
    proj = F.linear(x, attn.qkv_w.flatten(end_dim=1).type_as(x))
    print(f"proj shape: {proj.shape}")
    q, k, v = proj.view(B, 3 * attn.num_heads, T , attn.head_dim).chunk(3, dim=1)

    
    # Setup for compressed attention
    compress_divisible_seq_len = (T // attn.compress_block_size) * attn.compress_block_size
    num_compress_blocks = compress_divisible_seq_len // attn.compress_block_size
    
    # Setup for fine attention
    fine_divisible_seq_len = ((T + attn.selection_block_size - 1) // attn.selection_block_size) * attn.selection_block_size
    num_fine_blocks = fine_divisible_seq_len // attn.selection_block_size
    
    # Get memory components
    mem_ck, mem_cv = repeat(attn.compress_mem_kv, 'kv ... -> kv b ...', b=B)
    num_mem_compress_kv = mem_ck.shape[-2]
    
    # Step 1: Run compressed attention
    print("Running compressed attention...")
    compressed_attn_out, csim = attn.compressed_attention(
        q=q,
        k=k,
        v=v,
        mem_ck=mem_ck,
        mem_cv=mem_cv,
        num_mem_compress_kv=num_mem_compress_kv,
        num_compress_blocks=num_compress_blocks,
        compress_divisible_seq_len=compress_divisible_seq_len,
        device=x.device
    )
    print(f"Compressed attention output shape: {compressed_attn_out.shape}")
    print(f"Compressed similarity matrix shape: {csim.shape}")
    

    # Step 2: Apply rotary embeddings for fine attention
    q_rotary, k_rotary = attn.rotary(q), attn.rotary(k)
    
    # Run fine attention
    print("Running fine attention...")
    fine_attn_out = attn.fine_attention(
        fq=q_rotary,
        fk=k_rotary,
        fv=v,
        csim=csim,
        num_mem_compress_kv=num_mem_compress_kv,
        num_compress_blocks=num_compress_blocks,
        num_fine_blocks=num_fine_blocks,
        fine_divisible_seq_len=fine_divisible_seq_len,
        device=x.device
    )
    print(f"Fine attention output shape: {fine_attn_out.shape}")
    
    # Combine the outputs (just for demonstration)
    print("Combining outputs...")
    combined_output = 0.5 * compressed_attn_out + 0.5 * fine_attn_out
    combined_output = attn.merge_heads(combined_output)
    print(f"Combined output shape: {combined_output.shape}")
    
    return {
        "compressed_output": compressed_attn_out,
        "fine_output": fine_attn_out,
        "combined_output": combined_output
    }

def test_nsa_forward():
    """Test the full NSA_Attention forward pass"""
    # Parameters
    batch_size = 1
    seq_len = 1024
    model_dim = 768
    num_heads = 12
    head_dim = model_dim // num_heads
    sliding_window_size = 32
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Create NSA_Attention module
    attn = NSA_Attention(
        dim=model_dim,
        dim_head=head_dim,
        heads=num_heads,
        layer_idx=0,
        sliding_window_size=sliding_window_size,
        compress_block_size=16,
        selection_block_size=16,
        num_selected_blocks=16,
        query_heads_share_selected_kv=True,
        use_fine_flex_attention=False,
        use_triton_kernel=True,
        use_diff_topk=True
    ).to(device)
    
    # Create random input tensors
    x = torch.randn(batch_size, seq_len, model_dim).to(device)
    
    # Create a value embedding (can be None for testing)
    ve = None
    
    # Create a sliding window block mask for attention
    def sliding_window_mask_fn(b_idx, h_idx, q_idx, kv_idx):
        # Simple sliding window mask
        return abs(q_idx - kv_idx) <= sliding_window_size
    
    if torch.cuda.is_available():
        block_mask = create_block_mask(
            sliding_window_mask_fn, 
            B=batch_size, 
            H=num_heads, 
            Q_LEN=seq_len, 
            KV_LEN=seq_len, 
            _compile=True
        )
    else:
        block_mask = None
    
    # Run the full forward pass
    print("\nTesting full NSA_Attention forward pass...")
    output = attn.forward(x, ve, block_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape matches input shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    return {
        "input": x,
        "output": output
    }


def test_external_nsa():
    from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention
    import torch

    attn = SparseAttention(
        dim=768,
        dim_head=768 // 12,
        heads=12,
        sliding_window_size=32,
        compress_block_size=16,
        num_selected_blocks=16,
        selection_block_size=16,
        num_compressed_mem_kv=1,
        query_heads_share_selected_kv=True,
        use_triton_kernel=True,
        use_diff_topk=True,
    )

    device = torch.device("cuda")
    inp = torch.randn(1, 1024, 768).to(device)
    attn.to(device)
    attn.forward(inp)

if __name__ == "__main__":
    # Test individual components
    #component_results = test_nsa_components()
    #print("Component tests completed successfully!")
    test_external_nsa()
    # Test full forward pass
    forward_results = test_nsa_forward()
    print("Forward pass test completed successfully!")
