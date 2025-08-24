## Motivation
The current `torch.nn.attention.flex_attention` implementation only supports returning the logsumexp (LSE) of attention scores via the `return_lse` parameter. However, there are many other statistics a user might want to read  from the attention weights, that can be computed without materialising the entire attention matrix.
For example if we want to implement [MuonClip](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) we need access to the max per head attention logits, this is not possible with the current implementation of flex_attention, and users would have to write their own kernel, to obtain this information.

Thus i think it would be nice to allow users to access parts of QK^T matrix outside the kernel.

### API Design

One possible approach would be to allow the users to write a triton function specifying the reduction. And the dimensions to reduce over, IE kv, q, h, b. 

```python
@triton.jit 
def reduction(
    qk_ptr, # [BlockM, BlockN]
    accum_ptr, # [user specified dims]
    b_idx : int, 
    h_idx : int, 
    q_offs : int, 
    v_offs : int
): 
```

Or we might not want the user to write triton code but assume that the user only wants to compute a limited set of reductions like max, min, mean, sum.

```python
class ReductionType(Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    SUM = "sum"
```
