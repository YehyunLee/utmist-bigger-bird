"""GPU checks against independent dense/reference computations."""
import json
import torch
from kernels.bigger_bird_block_routing import block_metadata, select_distinct_blocks
from kernels.bigger_bird_routing import context_schedule
from kernels.bigger_bird_flash import bigger_bird_flash


def main():
    torch.manual_seed(49)
    q = torch.randn(2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    mask = torch.ones((1, 1024), device="cuda", dtype=torch.bool)
    mask[:, 96:110] = False
    rel, sal, anchors = block_metadata(q, k, 700, mask, 2, 128, 128)
    ref = (k.float() * q[:, 700:701].float()).sum(-1)
    valid = (torch.arange(1024, device="cuda") < 573)[None] & mask
    ref = ref.masked_fill(~valid, -float("inf")).reshape(2, 32, 32).amax(-1)
    torch.testing.assert_close(rel, ref, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(k.gather(1, anchors.clamp_min(0).long()[..., None].expand(-1, -1, 128)).float().mul(q[:, 700:701].float()).sum(-1)[torch.isfinite(ref)], ref[torch.isfinite(ref)], atol=3e-5, rtol=3e-5)
    sched = context_schedule(1024, budget_scaling="sqrt")
    routes, chunk = select_distinct_blocks(q, k, schedule=sched, token_mask=mask, num_heads=2, coverage_fraction=0.125)
    for head in routes[:, 0]:
        actual = head[head >= 0]
        assert actual.unique().numel() == actual.numel()
        assert actual.numel() == routes.shape[-1], (actual.numel(), routes.shape)
    # Changing a suffix cannot influence routing for an earlier fixed-layout
    # prefix group. This does not claim invariance when group layout changes.
    prefix_kw = dict(schedule=sched, num_heads=2, routing_mode="bounded_prefix", max_route_groups=4,
                     schedule_kwargs={"budget_scaling": "sqrt"}, coverage_fraction=0.125)
    before, prefix_chunk = select_distinct_blocks(q, k, **prefix_kw)
    q_changed, k_changed = q.clone(), k.clone()
    q_changed[:, 513:] = 100 * torch.randn_like(q_changed[:, 513:])
    k_changed[:, 513:] = 100 * torch.randn_like(k_changed[:, 513:])
    after, _ = select_distinct_blocks(q_changed, k_changed, **prefix_kw)
    assert torch.equal(before[:, :3], after[:, :3])
    for group in range(before.shape[1]):
        assert ((before[:, group] < 0) | (before[:, group] <= group * prefix_chunk)).all()
    # Sparse Flash output must equal an explicit masked softmax with exactly
    # the chosen union and the recent backbone, including partial tail blocks.
    qs = torch.randn(2, 129, 32, device="cuda", dtype=torch.bfloat16)
    ks, vs = torch.randn_like(qs), torch.randn_like(qs)
    sch = context_schedule(129, budget_scaling="sqrt")
    ix, width = select_distinct_blocks(qs, ks, schedule=sch, num_heads=2, recent_window=16, low_rank_dim=32, coverage_fraction=0.125)
    output = bigger_bird_flash(qs, ks, vs, ix, front=0, window=16, num_heads=2, route_chunk=width)
    positions = torch.arange(129, device="cuda")
    allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - 15)
    allowed = allowed[None].expand(2, -1, -1).clone()
    for head in range(2):
        selected = ix[head, 0]
        selected = selected[selected >= 0].long()
        allowed[head, :, selected] |= selected[None, :] <= positions[:, None]
    score = qs.float() @ ks.float().transpose(-1, -2)
    expected = score.masked_fill(~allowed, -float("inf")).softmax(-1) @ vs.float()
    torch.testing.assert_close(output.float(), expected, atol=0.04, rtol=0.04)
    # A block containing one unusually relevant token survives max scoring.
    needle_q = torch.zeros((1, 2048, 32), device="cuda", dtype=torch.bfloat16)
    needle_q[:, -1, 0] = 1
    needle_k = torch.zeros_like(needle_q)
    needle_k[:, 900, 0] = 20
    needle_ix, _ = select_distinct_blocks(needle_q, needle_k, schedule=context_schedule(2048, budget_scaling="sqrt"), num_heads=1, low_rank_dim=32, diversity=0, coverage_fraction=0.125)
    assert (needle_ix == 900).any()
    print(json.dumps({"gpu_checks": "PASS", "checks": ["block_max_matches_reference", "unique_full_budget", "prefix_future_independence_fixed_layout", "causal_route_edges", "sparse_flash_matches_masked_softmax", "single_token_signal_retained"]}), flush=True)


if __name__ == "__main__":
    main()
