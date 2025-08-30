from typing import Optional
import torch

def build_mask_mod(seg_ids, bound, root_seg_id, *args, **kwargs):
    """
    Build a mask for attention based on segment ids, bounds, and root segment id.
    """
    def mask_mod(b, h, q_idx, k_idx):
        # All inputs are broadcastable index tensors. We avoid advanced indexing on BatchedTensorImpl
        # and only use take_along_dim/gather along the LAST dimension of pre-shaped tensors.
        # Shapes (broadcasted):
        #   b:     [B, H, Q, K] (values 0..B-1)
        #   h:     [B, H, Q, K] (unused for logic, kept for shape)
        #   q_idx: [B, H, Q, K] (positions 0..S-1 along sequence)
        #   k_idx: [B, H, Q, K]
        # seg_ids: [B, S]
        # bound:   [B, num_segments]
        B, S = seg_ids.shape

        # Prepare tensors for take_along_dim along the last dim (=3)
        # seg_ids_exp: [B, 1, 1, S]
        seg_ids_exp = seg_ids.view(B, 1, 1, S)

        # Gather segment ids at q/k positions without per-element indexing
        # sq: [B, H, Q, K] but independent of K; we first get [B, 1, Q, 1] then broadcast
        sq = torch.take_along_dim(seg_ids_exp, q_idx.to(torch.long), dim=3)  # [B, H, Q, K] by broadcasting
        sk = torch.take_along_dim(seg_ids_exp, k_idx.to(torch.long), dim=3)  # [B, H, Q, K]

        # Padding mask: -1 denotes pad in seg_ids
        is_pad_q = sq.eq(-1)
        is_pad_k = sk.eq(-1)
        any_pad = is_pad_q | is_pad_k

        # Intra-segment causal: same segment and k<=q
        is_causal = k_idx <= q_idx
        intra = sq.eq(sk) & is_causal

        # Parent-allow attention: child segment may attend up to bound[b, sq] in the ROOT segment.
        # Gather per-(b, q) parent limit from `bound` using the segment id at q.
        # bound_exp: [B, 1, 1, num_segments]
        bound_exp = bound.view(B, 1, 1, bound.size(1))
        valid_q = ~is_pad_q  # only when sq != -1
        sq_safe = torch.where(valid_q, sq, torch.zeros_like(sq))  # clamp -1 -> 0 to avoid OOB
        parent_limit = torch.take_along_dim(bound_exp, sq_safe.to(torch.long), dim=3)  # [B, H, Q, K]

        # root_seg_id compare (broadcast to [B, H, Q, K])
        root_exp = root_seg_id.view(B, 1, 1, 1)
        is_root_k = sk.eq(root_exp)

        parent_allow = (k_idx <= parent_limit) & is_root_k & valid_q

        # Final mask (boolean)
        out = (intra | parent_allow) & is_causal & (~any_pad)
        return out

    # The rest of build_mask_mod implementation remains unchanged
    # ... (not shown as per instructions)
    pass