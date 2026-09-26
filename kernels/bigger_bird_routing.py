"""Parameter-free proposal-inspired local/global/long-range selection.

Global coverage uses content salience within temporal strata. Local and
long-range routing use one-step MMR on a relevance shortlist. This is an
inference variant of the proposal, not a trained gate or a facility-location
approximation guarantee. No random teleports or fixed front anchors.
"""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def context_schedule(n, *, middle_min=512, middle_max=2048, middle_ratio=64,
                     window_min=1024, window_max=8192, local_min=128,
                     local_max=512, globals_min=32, globals_max=64):
    # Parameters scale from the observed context; budgets remain capped.
    middle = min(middle_max, max(middle_min, math.ceil(n / middle_ratio)))
    wide = min(window_max, max(window_min, math.ceil(n / 16)))
    local = min(local_max, max(local_min, math.ceil(wide / 16)))
    glob = min(globals_max, max(globals_min, math.ceil(n / 2048)))
    return dict(middle=min(middle, n), wide=min(wide, n),
                local=min(local, n), globals=min(glob, n))


@triton.jit
def _scores(Q, K, MASK, REL, SAL,
            sq0: tl.constexpr, sq1: tl.constexpr, sq2: tl.constexpr,
            sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr,
            sm0: tl.constexpr, sm1: tl.constexpr,
            N: tl.constexpr, D: tl.constexpr, LOW: tl.constexpr,
            H: tl.constexpr, QPOS: tl.constexpr, QINDEX: tl.constexpr,
            HAS_MASK: tl.constexpr,
            BN: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(1)
    pos = tl.program_id(0) * BN + tl.arange(0, BN)
    ds = tl.arange(0, BD)
    q = tl.load(Q + bh * sq0 + QINDEX * sq1 + ds * sq2, ds < LOW, other=0).to(tl.float32)
    keys = tl.load(K + bh * sk0 + pos[:, None] * sk1 + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    prev = tl.load(K + bh * sk0 + tl.maximum(pos - 1, 0)[:, None] * sk1
                   + ds[None, :] * sk2,
                   (pos[:, None] < N) & (ds[None, :] < LOW), other=0).to(tl.float32)
    norm = tl.sum(keys * keys, 1)
    prev_norm = tl.sum(prev * prev, 1)
    novelty = 1.0 - tl.sum(keys * prev, 1) / tl.sqrt(tl.maximum(norm * prev_norm, 1.0e-12))
    relevance = tl.sum(keys * q[None, :], 1)
    salience = tl.sqrt(norm / LOW) + 0.25 * novelty
    valid = (pos < N) & (pos <= QPOS)
    if HAS_MASK:
        valid = valid & tl.load(MASK + (bh // H) * sm0 + pos * sm1, pos < N, other=0)
    tl.store(REL + bh * N + pos, tl.where(valid, relevance, -float("inf")), pos < N)
    tl.store(SAL + bh * N + pos, tl.where(valid, salience, -float("inf")), pos < N)


def content_scores(q, k, qpos, token_mask, num_heads, low_rank_dim):
    bh, n, d = k.shape
    low = min(d, low_rank_dim)
    rel = torch.empty((bh, n), dtype=torch.float32, device=q.device)
    sal = torch.empty_like(rel)
    _scores[(triton.cdiv(n, 64), bh)](
        q, k, token_mask if token_mask is not None else q, rel, sal,
        *q.stride(), *k.stride(),
        *(token_mask.stride() if token_mask is not None else (0, 0)),
        n, d, low, num_heads, qpos, qpos - (n - q.shape[1]), token_mask is not None,
        64, triton.next_power_of_2(low), num_warps=4)
    return rel, sal


@triton.jit
def _mmr_statistics(VALS, SPREAD, C: tl.constexpr, BC: tl.constexpr):
    bh = tl.program_id(0)
    slots = tl.arange(0, BC)
    vals = tl.load(VALS + bh * C + slots, slots < C, other=-float("inf"))
    finite = (slots < C) & (vals > -float("inf"))
    safe = tl.where(finite, vals, 0.0)
    count = tl.maximum(tl.sum(finite.to(tl.int32), 0), 1)
    mean = tl.sum(safe, 0) / count
    variance = tl.sum(tl.where(finite, (safe - mean) * (safe - mean), 0.0), 0) / count
    tl.store(SPREAD + bh, tl.maximum(tl.sqrt(variance), 1.0e-6))


@triton.jit
def _mmr_scores(K, VALS, IDX, SPREAD, ADJUSTED, sk0: tl.constexpr,
                sk1: tl.constexpr, sk2: tl.constexpr, C: tl.constexpr,
                LOW: tl.constexpr, DIVERSITY: tl.constexpr,
                BC: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(1)
    slots = tl.program_id(0) * BC + tl.arange(0, BC)
    ds = tl.arange(0, BD)
    vals = tl.load(VALS + bh * C + slots, slots < C, other=-float("inf"))
    idx = tl.load(IDX + bh * C + slots, slots < C, other=0)
    top = tl.load(IDX + bh * C)
    keys = tl.load(K + bh * sk0 + idx[:, None] * sk1 + ds[None, :] * sk2,
                   (slots[:, None] < C) & (ds[None, :] < LOW), other=0).to(tl.float32)
    best = tl.load(K + bh * sk0 + top * sk1 + ds * sk2,
                   ds < LOW, other=0).to(tl.float32)
    similarity = tl.sum(keys * best[None, :], 1) / tl.sqrt(tl.maximum(
        tl.sum(keys * keys, 1) * tl.sum(best * best, 0), 1.0e-12))
    spread = tl.load(SPREAD + bh)
    adjusted = vals - DIVERSITY * spread * similarity
    adjusted = tl.where(slots == 0, float("inf"), adjusted)
    tl.store(ADJUSTED + bh * C + slots, adjusted, slots < C)


def _mmr_shortlist(scores, keys, count, diversity, low_rank_dim):
    """One-step cosine MMR, analogous to exp 5's one-step local penalty.

Always preserves the best relevance candidate; remaining slots are reranked
against it. Work is on a 2*k shortlist, not a full N-by-N similarity matrix.
"""
    bh, n = scores.shape
    count = min(count, n)
    if count == 0:
        return torch.empty((bh, 0), device=scores.device, dtype=torch.int32)
    vals, candidates = scores.topk(min(n, 2 * count), dim=-1, sorted=True)
    if diversity > 0 and candidates.shape[-1] > 1:
        low = min(keys.shape[-1], low_rank_dim)
        adjusted = torch.empty_like(vals)
        spread = torch.empty((bh,), device=keys.device, dtype=torch.float32)
        _mmr_statistics[(bh,)](vals, spread, candidates.shape[-1],
            triton.next_power_of_2(candidates.shape[-1]), num_warps=4)
        _mmr_scores[(triton.cdiv(candidates.shape[-1], 64), bh)](keys, vals, candidates, spread, adjusted, *keys.stride(),
            candidates.shape[-1], low, diversity,
            64, triton.next_power_of_2(low), num_warps=4)
        order = adjusted.topk(count, dim=-1).indices
        chosen = torch.gather(candidates, 1, order)
        selected_values = torch.gather(vals, 1, order)
    else:
        chosen, selected_values = candidates[:, :count], vals[:, :count]
    return torch.where(torch.isfinite(selected_values), chosen, -1).to(torch.int32)


def _coverage_globals(salience, visible, count):
    """Choose the most salient valid token in each coverage stratum."""
    bh, n = salience.shape
    count = min(count, visible)
    if count == 0:
        return torch.empty((bh, 0), dtype=torch.int32, device=salience.device)
    width = triton.cdiv(visible, count)
    padded = F.pad(salience[:, :visible], (0, count * width - visible), value=-float("inf"))
    grouped = padded.reshape(bh, count, width)
    values, offsets = grouped.max(-1)
    positions = offsets + torch.arange(count, device=salience.device)[None, :] * width
    return torch.where(torch.isfinite(values), positions, -1).to(torch.int32)


def select_bigger_bird(q, k, *, schedule, token_mask=None, num_heads=32,
                      recent_window=128, diversity=0.05, low_rank_dim=128,
                      routing_mode="last_query", route_chunk=1024,
                      schedule_kwargs=None):
    """Return [BH,groups,K] unique indices and the routing group width.

    last_query reproduces the existing full-context routing convention, which
    is not prefix-invariant: cached generation has different semantics from
    rerouting the whole prompt at every step. causal_chunk
    uses only the first query and visible prefix of each group; it is a separate
    quality/speed experiment and obeys prefix invariance.
"""
    bh, n, _ = k.shape
    if q.shape[1] < n:
        if q.shape[1] != 1:
            raise ValueError("Cached routing currently supports one query token at a time")
        routing_mode = "last_query"
    if routing_mode not in ("last_query", "causal_chunk"):
        raise ValueError("routing_mode must be last_query or causal_chunk")
    chunk = max(64, triton.next_power_of_2(n)) if routing_mode == "last_query" else route_chunk
    if chunk % 32:
        raise ValueError("route_chunk must be a multiple of 32")
    rows = []
    max_selected = schedule["middle"] + schedule["local"] + schedule["globals"]
    for start in range(0, n, chunk):
        qpos = n - 1 if routing_mode == "last_query" else start
        rel, sal = content_scores(q, k, qpos, token_mask, num_heads, low_rank_dim)
        visible = qpos + 1
        current = context_schedule(visible, **(schedule_kwargs or {})) if routing_mode == "causal_chunk" else schedule
        wide_start = max(0, visible - current["wide"])
        recent_start = max(0, visible - recent_window)
        # Local filtering inside a growing neighborhood, excluding the exact
        # recent backbone already read directly by the attention kernel.
        local_scores = rel.clone()
        local_scores[:, :wide_start] = -float("inf")
        local_scores[:, recent_start:] = -float("inf")
        local = _mmr_shortlist(local_scores, k, current["local"], diversity, low_rank_dim)
        glob = _coverage_globals(sal, visible, current["globals"])
        # Content-biased long links replace uniform random teleports. Retain
        # the same relevance engine that worked in the previous NIAH runs.
        middle_scores = rel.clone()
        middle_scores[:, recent_start:] = -float("inf")
        middle = _mmr_shortlist(middle_scores, k, current["middle"], diversity, low_rank_dim)
        idx = torch.cat((glob, local, middle), -1).sort(-1).values
        duplicates = torch.zeros_like(idx, dtype=torch.bool)
        duplicates[:, 1:] = idx[:, 1:] == idx[:, :-1]
        idx = idx.masked_fill(duplicates, -1)
        # Fixed storage budget across groups, with -1 for unused entries.
        idx = F.pad(idx, (0, max_selected - idx.shape[-1]), value=-1)
        rows.append(idx)
    return torch.stack(rows, 1), chunk
