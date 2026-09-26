"""Proposal-inspired Bigger Bird with growing context and Triton FlashAttention.

Three content-aware branches: salience/coverage globals, MMR-filtered wide
local windows, and relevance/diversity long-range links. The selected union
has one joint softmax. A small exact recent backbone protects LM continuity.
This initial variant uses frozen weights and no learned gate.
"""
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from patches.llama.llama_patched_model import LlamaSparseAttention
from sparse_attn_utils import token_mask_1d
from kernels.bigger_bird_flash import bigger_bird_flash
from kernels.bigger_bird_routing import context_schedule, select_bigger_bird


class BiggerBirdFlashAttention(LlamaSparseAttention):
    def __init__(self, base_attn, *, middle_min=512, middle_max=2048,
                 middle_ratio=64, window_min=1024, window_max=8192,
                 local_min=128, local_max=512, globals_min=32,
                 globals_max=64, recent_window=128, diversity=0.05,
                 low_rank_dim=128, routing_mode="last_query",
                 route_chunk=1024, use_triton=True, block_m=64,
                 block_n=64):
        super().__init__(base_attn)
        if not use_triton:
            raise ValueError("This experiment requires the Triton FlashAttention backend")
        if min(middle_min, middle_max, middle_ratio, window_min, window_max,
               local_min, local_max, globals_min, globals_max, recent_window,
               low_rank_dim, route_chunk) < 1:
            raise ValueError("All context/budget parameters must be positive")
        if middle_min > middle_max or window_min > window_max or local_min > local_max or globals_min > globals_max:
            raise ValueError("Minimum budget/window values must not exceed maxima")
        if diversity < 0 or routing_mode not in ("last_query", "causal_chunk"):
            raise ValueError("Invalid diversity or routing mode")
        if routing_mode == "causal_chunk" and route_chunk % block_m:
            raise ValueError("Causal route chunks must align with the query tile size")
        self.schedule_kwargs = dict(middle_min=middle_min, middle_max=middle_max,
            middle_ratio=middle_ratio, window_min=window_min, window_max=window_max,
            local_min=local_min, local_max=local_max,
            globals_min=globals_min, globals_max=globals_max)
        self.recent_window = recent_window
        self.diversity = diversity
        self.low_rank_dim = low_rank_dim
        self.routing_mode = routing_mode
        self.route_chunk = route_chunk
        self.block_m = block_m
        self.block_n = block_n
        self.last_diagnostics = {}
        self._cached_routes = None
        self._cached_anchor = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        bsz, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        # Store unexpanded GQA keys/values. Cache retention is independent of
        # sparse attention selection: unselected tokens remain available.
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        src_len = k.shape[-2]
        k = k.repeat_interleave(self.num_kv_groups, 1)
        v = v.repeat_interleave(self.num_kv_groups, 1)
        bh = bsz * self.num_heads
        q = q.reshape(bh, length, self.head_dim) * self.scaling
        k = k.reshape(bh, src_len, self.head_dim)
        v = v.reshape(bh, src_len, self.head_dim)
        # HF additive causal masks use zero for allowed edges and negative
        # values for blocked edges; the historical helper expects booleans.
        if attention_mask is not None and attention_mask.ndim >= 3 and attention_mask.is_floating_point():
            attention_mask = attention_mask >= 0
        token_mask = token_mask_1d(attention_mask, bsz, src_len, q.device)
        out = self.sparse_attention(q, k, v, token_mask, bsz, self.num_heads,
                                    is_causal=self.is_causal)
        out = out.view(bsz, self.num_heads, length, self.head_dim).transpose(1, 2).reshape(bsz, length, -1)
        return self.o_proj(out), None

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if self.training or not is_causal:
            raise ValueError("Bigger Bird v2 currently supports causal inference only")
        n = K.shape[1]
        decode = Q.shape[1] < n
        anchor = ((n - 1) // self.route_chunk) * self.route_chunk
        schedule_n = anchor + 1 if self.routing_mode == "causal_chunk" else n
        actual_schedule = context_schedule(schedule_n, **self.schedule_kwargs)
        reused = (decode and self.routing_mode == "causal_chunk"
                  and self._cached_anchor == anchor and self._cached_routes is not None)
        if reused:
            indices = self._cached_routes
            chunk = max(64, 1 << (n - 1).bit_length())
        else:
            schedule = context_schedule(n, **self.schedule_kwargs)
            indices, chunk = select_bigger_bird(Q, K, schedule=schedule,
                token_mask=token_mask, num_heads=num_heads,
                recent_window=self.recent_window, diversity=self.diversity,
                low_rank_dim=self.low_rank_dim, routing_mode=self.routing_mode,
                route_chunk=self.route_chunk, schedule_kwargs=self.schedule_kwargs)
        self._cached_routes = indices[:, -1:, :].detach().clone()
        self._cached_anchor = anchor
        self.last_diagnostics = dict(actual_schedule, recent_window=self.recent_window,
            routing_mode=self.routing_mode, route_chunk=chunk, configured_route_chunk=self.route_chunk,
            prefix_invariant=self.routing_mode == "causal_chunk",
            backend="triton_sparse_flash", selection="content_globals+local_mmr+long_range_mmr",
            route_slots=indices.shape[-1], block_m=self.block_m, block_n=self.block_n)
        self.last_diagnostics.update(query_tokens=Q.shape[1], source_tokens=n,
            cached_decode=decode, reused_routes=reused,
            decode_routing="prefix_chunk_reuse" if reused else ("current_query" if decode else self.routing_mode))
        self.last_selected_indices = self._cached_routes[:, 0, :]
        return bigger_bird_flash(Q, K, V, indices, front=0,
            window=self.recent_window, token_mask=token_mask,
            num_heads=num_heads, route_chunk=chunk, scale=1.0,
            block_m=self.block_m, block_n=self.block_n)
