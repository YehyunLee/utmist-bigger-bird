"""Question-conditioned block routing and separately labeled oracle diagnostics.

Ordinary routes use only queries at natural-language question positions plus
the latest query. Gold positions enter only force_diagnostic_span, which is
never enabled for benchmark candidates. Actual attention keeps rotated Q/K.
"""
import torch
import triton
import triton.language as tl
from kernels.bigger_bird_block_routing import _exclude, _coverage_blocks, _expand_blocks
from kernels.bigger_bird_routing import _mmr_shortlist


@triton.jit
def _question_scores(Q, K, MASK, REL, SAL, ANCHOR,
                     sq0: tl.constexpr, sq1: tl.constexpr, sq2: tl.constexpr,
                     sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr,
                     sm0: tl.constexpr, sm1: tl.constexpr,
                     N: tl.constexpr, NB: tl.constexpr, LOW: tl.constexpr,
                     H: tl.constexpr, R: tl.constexpr, RECENT: tl.constexpr,
                     HAS_MASK: tl.constexpr, BD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    pos = block * 32 + tl.arange(0, 32)
    ds = tl.arange(0, BD)
    keys = tl.load(K + bh * sk0 + pos[:, None] * sk1 + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    prev = tl.load(K + bh * sk0 + tl.maximum(pos - 1, 0)[:, None] * sk1 + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    knorm = tl.sum(keys * keys, 1)
    best_token = tl.full((32,), -float("inf"), tl.float32)
    for r in range(R):
        query = tl.load(Q + bh * sq0 + r * sq1 + ds * sq2, ds < LOW, other=0).to(tl.float32)
        qnorm = tl.sum(query * query, 0)
        score = tl.sum(keys * query[None, :], 1) / tl.sqrt(tl.maximum(knorm * qnorm, 1.0e-12))
        best_token = tl.maximum(best_token, score)
    valid = (pos < N - RECENT)
    if HAS_MASK:
        valid = valid & tl.load(MASK + (bh // H) * sm0 + pos * sm1, pos < N, other=0)
    relevance = tl.where(valid, best_token, -float("inf"))
    novelty = 1 - tl.sum(keys * prev, 1) / tl.sqrt(tl.maximum(knorm * tl.sum(prev * prev, 1), 1.0e-12))
    salience = tl.where(valid, tl.sqrt(knorm / LOW) + 0.25 * novelty, -float("inf"))
    best = tl.max(relevance, 0)
    anchor = tl.min(tl.where(valid & (relevance == best), pos, N), 0)
    tl.store(REL + bh * NB + block, best)
    tl.store(SAL + bh * NB + block, tl.max(salience, 0))
    tl.store(ANCHOR + bh * NB + block, tl.where(anchor < N, anchor, -1))


def question_metadata(q, k, query_positions, token_mask, num_heads, low_rank_dim, recent_window):
    bh, n, d = k.shape
    assert q.shape == k.shape and query_positions and all(0 <= p < n for p in query_positions)
    positions = torch.tensor(query_positions, device=q.device, dtype=torch.long)
    queries = q.index_select(1, positions).contiguous()
    nb, low = triton.cdiv(n, 32), min(d, low_rank_dim)
    rel = torch.empty((bh, nb), device=k.device, dtype=torch.float32)
    sal, anchors = torch.empty_like(rel), torch.empty((bh, nb), device=k.device, dtype=torch.int32)
    _question_scores[(nb, bh)](queries, k, token_mask if token_mask is not None else q,
        rel, sal, anchors, *queries.stride(), *k.stride(),
        *(token_mask.stride() if token_mask is not None else (0, 0)),
        n, nb, low, num_heads, len(query_positions), recent_window, token_mask is not None,
        triton.next_power_of_2(low), num_warps=4)
    return rel, sal, anchors


def select_question_blocks(q, k, *, schedule, query_positions, token_mask=None,
                           num_heads=32, recent_window=128, diversity=0.05,
                           low_rank_dim=128, coverage_fraction=0.125):
    bh, n, d = k.shape
    budget = min(triton.cdiv(sum(schedule[x] for x in ("middle", "local", "globals")), 32), triton.cdiv(n, 32))
    rel, sal, anchors = question_metadata(q, k, query_positions, token_mask, num_heads, low_rank_dim, recent_window)
    global_count = min(max(2, round(budget * coverage_fraction)), budget)
    local_count = min(max(1, schedule["local"] // 32), budget - global_count)
    glob = _coverage_blocks(sal, triton.cdiv(n, 32), global_count)
    remaining = _exclude(rel, glob)
    block_keys = k.gather(1, anchors.clamp_min(0).long()[..., None].expand(-1, -1, d))
    local_scores = remaining.clone()
    local_scores[:, :max(0, n - schedule["wide"]) // 32] = -float("inf")
    local = _mmr_shortlist(local_scores, block_keys, local_count, diversity, low_rank_dim)
    remaining = _exclude(remaining, local)
    reserved = torch.cat((glob, local), -1)
    middle = _mmr_shortlist(remaining, block_keys, budget, diversity, low_rank_dim)
    middle = torch.where(torch.arange(middle.shape[-1], device=k.device)[None] < (budget - (reserved >= 0).sum(-1))[:, None], middle, -1)
    chosen = torch.cat((reserved, middle), -1)
    chosen = torch.where(chosen >= 0, chosen, rel.shape[-1]).sort(-1).values[:, :budget]
    chosen = torch.where(chosen < rel.shape[-1], chosen, -1).contiguous()
    expanded = torch.empty((bh, budget * 32), device=k.device, dtype=torch.int32)
    _expand_blocks[(triton.cdiv(budget * 32, 128), bh)](chosen, expanded,
        *chosen.stride(), chosen.shape[-1], n, n, budget * 32, 32, 128, num_warps=4)
    return expanded[:, None, :], max(64, triton.next_power_of_2(n))


def force_diagnostic_span(indices, span, n):
    """Insert exactly this gold-aware diagnostic context, with no budget growth.

    Input routes must already be unique. Invalid tail slots remain invalid.
    This function is explicitly excluded from all ordinary candidate paths.
    """
    begin, end = span
    assert 0 <= begin < end <= n and end - begin <= indices.shape[-1]
    old = indices.masked_fill((indices >= begin) & (indices < end), -1)
    old = torch.where(old >= 0, old, n).sort(-1).values
    forced = torch.arange(begin, end, device=indices.device, dtype=indices.dtype)
    forced = forced.expand(*indices.shape[:-1], -1)
    union = torch.cat((forced, old), -1)[..., :indices.shape[-1]].sort(-1).values
    return torch.where(union < n, union, -1).contiguous()
