"""GPU correctness checks against an independent unique-union reference."""
import json
import torch
from kernels.bigger_bird_flash import bigger_bird_flash
from kernels.bigger_bird_routing import context_schedule, select_bigger_bird, _mmr_shortlist


def reference(q, k, v, idx, window, front, chunk, padding=None, heads=2):
    bh, n, _ = q.shape
    pos = torch.arange(n, device=q.device)
    local = (pos[None, :] <= pos[:, None]) & (pos[None, :] >= pos[:, None] - window + 1)
    anchors = (pos[None, :] < front) & (pos[None, :] <= pos[:, None])
    allowed = (local | anchors)[None].expand(bh, -1, -1).clone()
    for group in range(idx.shape[1]):
        start, end = group * chunk, min(n, (group + 1) * chunk)
        for head in range(bh):
            keys = idx[head, group]
            keys = keys[keys >= 0]
            allowed[head, start:end, keys] = True
    allowed &= pos[None, None, :] <= pos[None, :, None]
    if padding is not None:
        allowed &= padding.repeat_interleave(heads, 0)[:, None, :]
    scores = q.float() @ k.float().transpose(-1, -2)
    scores.masked_fill_(~allowed, -float("inf"))
    weights = torch.softmax(scores, -1).nan_to_num()
    out = weights @ v.float()
    if padding is not None:
        out *= padding.repeat_interleave(heads, 0)[:, :, None]
    return out


def main():
    torch.manual_seed(53)
    # Verify the fused diversity reranker against its independent PyTorch formula.
    keys = torch.randn(2, 257, 128, dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(2, 257, device="cuda")
    scores[:, -17:] = -float("inf")
    vals, candidates = scores.topk(128, sorted=True)
    gathered = torch.gather(keys, 1, candidates[..., None].expand(-1, -1, 128)).float()
    normalized = torch.nn.functional.normalize(gathered, dim=-1)
    adjusted = vals - 0.05 * vals.std(-1, correction=0, keepdim=True) * (normalized * normalized[:, :1]).sum(-1)
    adjusted[:, 0] = float("inf")
    expected = torch.gather(candidates, 1, adjusted.topk(64).indices)
    actual = _mmr_shortlist(scores, keys, 64, 0.05, 128)
    assert torch.equal(actual.sort(-1).values.long(), expected.sort(-1).values)
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for n, d, window, front, chunk, masked in ((17, 64, 8, 0, 64, False), (97, 64, 17, 7, 128, False),
                (257, 128, 33, 0, 128, True), (513, 128, 128, 0, 1024, False)):
            q = torch.randn(4, n, d, device="cuda", dtype=dtype) / d**0.5
            k, v = torch.randn_like(q), torch.randn_like(q)
            idx = torch.stack([torch.randperm(n, device="cuda")[:min(63, n)] for _ in range(4 * ((n + chunk - 1)//chunk))]).reshape(4, -1, min(63,n)).to(torch.int32)
            padding = torch.ones(2, n, dtype=torch.bool, device="cuda") if masked else None
            if padding is not None:
                padding[0, 15:23] = False
                padding[1, -17:] = False
            ref = reference(q, k, v, idx, window, front, chunk, padding, heads=2)
            fast = bigger_bird_flash(q, k, v, idx, window=window, front=front,
                route_chunk=chunk, token_mask=padding, num_heads=2, block_m=32)
            error = (ref - fast.float()).abs()
            item = dict(dtype=str(dtype), n=n, d=d, padding=masked,
                        max_error=error.max().item(), mean_error=error.mean().item())
            print(json.dumps(item), flush=True)
            assert torch.isfinite(fast).all(), item
            torch.testing.assert_close(fast.float(), ref, atol=0.025 if dtype == torch.bfloat16 else 0.004, rtol=0.025)
            wide_tile = bigger_bird_flash(q, k, v, idx, window=window, front=front,
                route_chunk=chunk, token_mask=padding, num_heads=2, block_m=64)
            torch.testing.assert_close(wide_tile.float(), ref,
                atol=0.025 if dtype == torch.bfloat16 else 0.004, rtol=0.025)
            results.append(item)
    # Router output must be unique and padding/future-safe; causal mode
    # must also preserve the same output after a suffix is appended.
    q = torch.randn(2, 257, 128, device="cuda", dtype=torch.bfloat16) / 128**0.5
    k, v = torch.randn_like(q), torch.randn_like(q)
    for mode in ("last_query", "causal_chunk"):
        idx, chunk = select_bigger_bird(q, k, schedule=context_schedule(257),
            num_heads=2, routing_mode=mode, route_chunk=128)
        for row in idx.flatten(0, 1):
            valid = row[row >= 0]
            assert valid.unique().numel() == valid.numel()
        ref = reference(q, k, v, idx, 128, 0, chunk)
        fast = bigger_bird_flash(q, k, v, idx, front=0, window=128,
                                route_chunk=chunk, num_heads=2)
        torch.testing.assert_close(fast.float(), ref, atol=0.025, rtol=0.025)
        if mode == "last_query":
            # Unequal Q/K lengths must use original absolute positions, not
            # the packed sequence's default triangular causal alignment.
            cached_idx, cached_chunk = select_bigger_bird(q[:, -1:], k,
                schedule=context_schedule(257), num_heads=2,
                routing_mode=mode, route_chunk=128)
            cached = bigger_bird_flash(q[:, -1:], k, v, cached_idx, front=0,
                window=128, route_chunk=cached_chunk, num_heads=2)
            torch.testing.assert_close(cached, fast[:, -1:], atol=0.005, rtol=0.005)
        if mode == "causal_chunk":
            prefix = 193
            shorter, short_chunk = select_bigger_bird(q[:, :prefix], k[:, :prefix],
                schedule=context_schedule(prefix), num_heads=2,
                routing_mode=mode, route_chunk=128)
            early = bigger_bird_flash(q[:, :prefix], k[:, :prefix], v[:, :prefix],
                shorter, front=0, window=128, route_chunk=short_chunk, num_heads=2)
            torch.testing.assert_close(early, fast[:, :prefix], atol=0.005, rtol=0.005)
    print("BIGGER_BIRD_FLASH_CORRECTNESS_PASS", flush=True)


if __name__ == "__main__":
    main()
