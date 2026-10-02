"""Uncached research attention; all benchmark layers remain sparse.

Question-conditioned proxies may be taken before positional rotation. Only
routing changes; the sparse Flash kernel always uses correctly rotated Q/K.
Gold-aware diagnostic arms are marked and cannot use ordinary run metadata.
"""
from functools import partial
from kernels.bigger_bird_routing import context_schedule
from kernels.bigger_bird_block_routing import select_distinct_blocks
from kernels.bigger_bird_query_routing import select_question_blocks, force_diagnostic_span
from kernels.bigger_bird_flash import bigger_bird_flash
from experiments.exp_19_bigger_bird_flash.model_llama import BiggerBirdFlashAttention


class InvestigationAttention(BiggerBirdFlashAttention):
    def __init__(self, base_attn, *, query_proxy="coverage", diagnostic_oracle="none", **kwargs):
        super().__init__(base_attn, **kwargs)
        assert query_proxy in ("coverage", "rotated_question", "raw_question")
        assert diagnostic_oracle in ("none", "all", "late8")
        assert diagnostic_oracle == "none" or query_proxy == "coverage"
        self.query_proxy, self.diagnostic_oracle = query_proxy, diagnostic_oracle
        self.question_span = None
        self.diagnostic_span = None
        self._raw_q = self._raw_k = None

    def forward(self, hidden_states, *args, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Round2 is validated only without KV caching")
        hooks = []
        if self.query_proxy == "raw_question":
            hooks = [self.q_proj.register_forward_hook(lambda m, a, o: setattr(self, "_raw_q", o)),
                     self.k_proj.register_forward_hook(lambda m, a, o: setattr(self, "_raw_k", o))]
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
            self._raw_q = self._raw_k = None

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if self.training or not is_causal or Q.shape[1] != K.shape[1]:
            raise ValueError("Round2 supports causal uncached inference only")
        n = K.shape[1]
        schedule = context_schedule(n, **self.schedule_kwargs)
        if self.query_proxy == "coverage":
            indices, chunk = select_distinct_blocks(Q, K, schedule=schedule,
                num_heads=num_heads, token_mask=token_mask, recent_window=self.recent_window,
                diversity=self.diversity, low_rank_dim=self.low_rank_dim, coverage_fraction=0.125)
        else:
            assert self.question_span is not None and self.diagnostic_span is None
            begin, end = self.question_span
            assert 0 <= begin < end <= n
            # Up to eight evenly spaced natural-question queries plus latest
            # token. Positions depend on question text, never its gold answer.
            count = min(8, end - begin)
            positions = sorted(set(begin + round(i * (end - begin - 1) / max(1, count - 1)) for i in range(count)) | {n - 1})
            route_q, route_k = Q, K
            if self.query_proxy == "raw_question":
                route_q = self._raw_q.view(bsz, n, num_heads, self.head_dim).transpose(1, 2).reshape(bsz * num_heads, n, self.head_dim)
                route_k = self._raw_k.view(bsz, n, self.num_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.num_kv_groups, 1).reshape(bsz * num_heads, n, self.head_dim)
            indices, chunk = select_question_blocks(route_q, route_k, schedule=schedule,
                query_positions=positions, num_heads=num_heads, token_mask=token_mask,
                recent_window=self.recent_window, diversity=self.diversity, low_rank_dim=self.low_rank_dim)
        oracle_applied = self.diagnostic_oracle == "all" or (self.diagnostic_oracle == "late8" and self.layer_idx >= 24)
        if self.diagnostic_oracle != "none":
            assert self.diagnostic_span is not None and token_mask is None or self.diagnostic_span is not None and bool(token_mask.all())
            if oracle_applied:
                indices = force_diagnostic_span(indices, self.diagnostic_span, n)
        else:
            assert self.diagnostic_span is None, "Gold offsets prohibited for ordinary selectors"
        self.last_selected_indices = indices[:, 0].detach()
        self.last_diagnostics = dict(schedule, route_slots=indices.shape[-1], recent_window=self.recent_window,
            query_proxy=self.query_proxy, diagnostic_oracle=self.diagnostic_oracle, oracle_applied=oracle_applied,
            backend="triton_sparse_flash", prefix_invariant=False)
        return bigger_bird_flash(Q, K, V, indices, front=0, window=self.recent_window,
            token_mask=token_mask, num_heads=num_heads, route_chunk=chunk, scale=1,
            block_m=self.block_m, block_n=self.block_n)
