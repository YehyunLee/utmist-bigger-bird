"""Dense-layer interventions for a teacher-forced long-context diagnostic."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.exp_19_bigger_bird_flash.model_investigation import InvestigationAttention


class Round3ProbeAttention(InvestigationAttention):
    def __init__(self, base_attn, *, dense_layers=(), **kwargs):
        super().__init__(base_attn, **kwargs)
        self.dense_layers = frozenset(int(x) for x in dense_layers)
        self.last_selected_indices = None

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if self.layer_idx not in self.dense_layers:
            return super().sparse_attention(Q, K, V, token_mask, bsz, num_heads, is_causal)
        if self.training or not is_causal or Q.shape[1] != K.shape[1]:
            raise ValueError("Round3 dense-layer control requires causal uncached prefill")
        if token_mask is not None and not bool(token_mask.all()):
            raise ValueError("Round3 control requires an unpadded single prompt")
        n, dim = Q.shape[1:]
        q = Q.view(bsz, num_heads, n, dim)
        k = K.view(bsz, num_heads, n, dim)
        v = V.view(bsz, num_heads, n, dim)
        # Q is already multiplied by the model's attention scale.  Use scale=1
        # so this is the same causal operation as exp0's FlashAttention path.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=1.0
            )
        self.last_selected_indices = None
        return out.reshape(bsz * num_heads, n, dim)
