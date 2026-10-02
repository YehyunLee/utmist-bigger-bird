"""Experimental distinct-span routing; frozen weights, no answer-aware routing.

Each 32-token block receives its maximum valid token relevance, rather than
the mean of its keys. Content coverage, wide-local MMR, and long-range MMR
select disjoint blocks. The existing Triton sparse Flash kernel reads the
union together with the exact recent backbone using one softmax.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from kernels.bigger_bird_routing import context_schedule, _mmr_shortlist


@triton.jit
def _block_metadata(Q, K, MASK, REL, SAL, ANCHOR,
                    sq0: tl.constexpr, sq1: tl.constexpr, sq2: tl.constexpr,
                    sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr,
                    sm0: tl.constexpr, sm1: tl.constexpr,
                    N: tl.constexpr, NB: tl.constexpr, LOW: tl.constexpr,
                    H: tl.constexpr, QPOS: tl.constexpr, RECENT: tl.constexpr,
                    HAS_MASK: tl.constexpr, BLOCK: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    pos = block * BLOCK + tl.arange(0, BLOCK)
    ds = tl.arange(0, BD)
    q = tl.load(Q + bh * sq0 + ds * sq2, ds < LOW, other=0).to(tl.float32)
    keys = tl.load(K + bh * sk0 + pos[:, None] * sk1 + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    prev = tl.load(K + bh * sk0 + tl.maximum(pos - 1, 0)[:, None] * sk1 + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    valid = (pos < N) & (pos <= QPOS) & (pos < QPOS + 1 - RECENT)
    if HAS_MASK:
        valid = valid & tl.load(MASK + (bh // H) * sm0 + pos * sm1, pos < N, other=0)
    norm = tl.sum(keys * keys, 1)
    previous_norm = tl.sum(prev * prev, 1)
    novelty = 1.0 - tl.sum(keys * prev, 1) / tl.sqrt(tl.maximum(norm * previous_norm, 1.0e-12))
    relevance = tl.where(valid, tl.sum(keys * q[None, :], 1), -float("inf"))
    salience = tl.where(valid, tl.sqrt(norm / LOW) + 0.25 * novelty, -float("inf"))
    best = tl.max(relevance, 0)
    anchor = tl.min(tl.where(valid & (relevance == best), pos, N), 0)
    tl.store(REL + bh * NB + block, best)
    tl.store(SAL + bh * NB + block, tl.max(salience, 0))
    tl.store(ANCHOR + bh * NB + block, tl.where(anchor < N, anchor, -1))


@triton.jit
def _expand_blocks(BLOCKS, OUT, sb0: tl.constexpr, sb1: tl.constexpr,
                   COUNT: tl.constexpr, N: tl.constexpr, VISIBLE: tl.constexpr,
                   BUDGET: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr):
    bh = tl.program_id(1)
    slot = tl.program_id(0) * TILE + tl.arange(0, TILE)
    index = tl.load(BLOCKS + bh * sb0 + (slot // BLOCK) * sb1,
                    (slot < BUDGET) & (slot // BLOCK < COUNT), other=-1)
    pos = index * BLOCK + slot % BLOCK
    valid = (index >= 0) & (pos < N) & (pos < VISIBLE)
    tl.store(OUT + bh * BUDGET + slot, tl.where(valid, pos, -1), slot < BUDGET)


def block_metadata(q, k, qpos, token_mask, num_heads, low_rank_dim, recent_window, query_window=1):
    bh, n, d = k.shape
    block = 32
    nb = triton.cdiv(n, block)
    low = min(d, low_rank_dim)
    qi = qpos - (n - q.shape[1])
    if not 0 <= qi < q.shape[1]:
        raise ValueError("Routing query must be represented in Q")
    qs = q[:, max(0, qi + 1 - query_window):qi + 1, :low].float().mean(1, keepdim=True).to(q.dtype)
    rel = torch.empty((bh, nb), device=k.device, dtype=torch.float32)
    sal = torch.empty_like(rel)
    anchors = torch.empty((bh, nb), device=k.device, dtype=torch.int32)
    _block_metadata[(nb, bh)](qs, k, token_mask if token_mask is not None else q,
        rel, sal, anchors, *qs.stride(), *k.stride(),
        *(token_mask.stride() if token_mask is not None else (0, 0)),
        n, nb, low, num_heads, qpos, recent_window, token_mask is not None,
        block, triton.next_power_of_2(low), num_warps=4)
    return rel, sal, anchors


def _exclude(scores, selected):
    used = torch.zeros(scores.shape, device=scores.device, dtype=torch.int32)
    used.scatter_add_(1, selected.clamp_min(0).long(), (selected >= 0).to(torch.int32))
    return scores.masked_fill(used > 0, -float("inf"))


def _coverage_blocks(salience, visible_blocks, count):
    count = min(count, visible_blocks)
    if count == 0:
        return torch.empty((salience.shape[0], 0), device=salience.device, dtype=torch.int32)
    width = triton.cdiv(visible_blocks, count)
    grouped = F.pad(salience[:, :visible_blocks], (0, count * width - visible_blocks), value=-float("inf"))
    values, offsets = grouped.reshape(salience.shape[0], count, width).max(-1)
    ids = offsets + torch.arange(count, device=salience.device)[None, :] * width
    return torch.where(torch.isfinite(values), ids, -1).to(torch.int32)


def select_distinct_blocks(q, k, *, schedule, token_mask=None, num_heads=32,
                           recent_window=128, diversity=0.05, low_rank_dim=128,
                           routing_mode="last_query", route_chunk=1024,
                           schedule_kwargs=None, query_window=1, max_route_groups=8,
                           coverage_fraction=None):
    bh, n, d = k.shape
    if q.shape[1] != n:
        raise ValueError("This R&D selector is tested only with uncached full-context generation")
    if routing_mode not in ("last_query", "bounded_prefix"):
        raise ValueError("R&D supports last_query or bounded_prefix")
    chunk = max(64, triton.next_power_of_2(n)) if routing_mode == "last_query" else max(64, triton.next_power_of_2(triton.cdiv(n, max_route_groups)))
    max_budget = sum(schedule[key] for key in ("middle", "local", "globals"))
    max_budget = triton.cdiv(max_budget, 32) * 32
    rows = []
    for start in range(0, n, chunk):
        qpos = n - 1 if routing_mode == "last_query" else start
        current = schedule if routing_mode == "last_query" else context_schedule(qpos + 1, **(schedule_kwargs or {}))
        budget = triton.cdiv(sum(current[key] for key in ("middle", "local", "globals")), 32)
        rel, sal, anchors = block_metadata(q, k, qpos, token_mask, num_heads, low_rank_dim, recent_window, query_window)
        visible_blocks = triton.cdiv(qpos + 1, 32)
        budget = min(budget, visible_blocks)
        global_count = max(1, current["globals"] // 32) if coverage_fraction is None else max(2, int(round(budget * coverage_fraction)))
        global_count = min(global_count, budget)
        local_count = min(max(1, current["local"] // 32), budget - global_count)
        globals_ = _coverage_blocks(sal, visible_blocks, global_count)
        remaining = _exclude(rel, globals_)
        # Representative key is the strongest query-relevant token in each
        # block; averaging all keys would dilute short retrieval evidence.
        block_keys = k.gather(1, anchors.clamp_min(0).long()[..., None].expand(-1, -1, d))
        local_scores = remaining.clone()
        wide_start_block = max(0, qpos + 1 - current["wide"]) // 32
        local_scores[:, :wide_start_block] = -float("inf")
        local = _mmr_shortlist(local_scores, block_keys, local_count, diversity, low_rank_dim)
        remaining = _exclude(remaining, local)
        # Refill unused coverage/local slots with relevance-ranked distinct
        # blocks. A per-head quota avoids device-to-host synchronization.
        reserved = torch.cat((globals_, local), -1)
        reserved_valid = (reserved >= 0).sum(-1)
        middle = _mmr_shortlist(remaining, block_keys, budget, diversity, low_rank_dim)
        middle = torch.where(torch.arange(middle.shape[-1], device=k.device)[None, :] < (budget - reserved_valid)[:, None], middle, -1)
        chosen = torch.cat((reserved, middle), -1)
        # Compact invalid entries before expansion; branch selections are
        # already disjoint, so no expanded-token sort/dedup is needed.
        chosen = torch.where(chosen >= 0, chosen, torch.full_like(chosen, rel.shape[-1]))
        chosen = chosen.sort(-1).values[:, :budget]
        chosen = torch.where(chosen < rel.shape[-1], chosen, -1).contiguous()
        expanded = torch.empty((bh, max_budget), device=k.device, dtype=torch.int32)
        _expand_blocks[(triton.cdiv(max_budget, 128), bh)](chosen, expanded,
            *chosen.stride(), chosen.shape[-1], n, qpos + 1, max_budget, 32, 128, num_warps=4)
        rows.append(expanded)
    return torch.stack(rows, 1), chunk


def legacy_spans32(original_selector):
    """Exact preserved span wrapper for the control arm."""
    def selector(*args, **kw):
        schedule = kw["schedule"]
        reduced = dict(schedule)
        for key in ("middle", "local", "globals"):
            reduced[key] = max(1, schedule[key] // 32)
        kw["schedule"] = reduced
        indices, chunk = original_selector(*args, **kw)
        bh, groups, _ = indices.shape
        n = (args[1] if len(args) > 1 else kw["k"]).shape[1]
        expanded = (indices.clamp_min(0) // 32)[..., None] * 32 + torch.arange(32, device=indices.device, dtype=indices.dtype)
        expanded = torch.where((indices[..., None] >= 0) & (expanded < n), expanded, -1).reshape(bh, groups, -1).sort(-1).values
        duplicate = torch.zeros_like(expanded, dtype=torch.bool)
        duplicate[..., 1:] = (expanded[..., 1:] == expanded[..., :-1]) & (expanded[..., 1:] >= 0)
        expanded = expanded.masked_fill(duplicate, -1)
        budget = sum(schedule[key] for key in ("middle", "local", "globals"))
        expanded = expanded[..., :budget]
        if expanded.shape[-1] < budget:
            expanded = F.pad(expanded, (0, budget - expanded.shape[-1]), value=-1)
        return expanded, chunk
    return selector
