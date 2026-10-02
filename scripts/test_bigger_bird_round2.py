"""Independent GPU references for question proxy, oracle separation and attention."""
import json
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from kernels.bigger_bird_query_routing import question_metadata, select_question_blocks, force_diagnostic_span
from kernels.bigger_bird_routing import context_schedule
from experiments.exp_19_bigger_bird_flash.model_investigation import InvestigationAttention


def main():
    torch.manual_seed(93)
    q = torch.randn(2, 1009, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    mask = torch.ones((1, 1009), device="cuda", dtype=torch.bool)
    mask[:, 101:115] = False
    positions = [980, 987, 1008]
    rel, sal, anchors = question_metadata(q, k, positions, mask, 2, 32, 128)
    scores = F.normalize(q[:, positions].float(), dim=-1) @ F.normalize(k.float(), dim=-1).transpose(-1, -2)
    valid = (torch.arange(1009, device="cuda") < 881)[None] & mask
    token_scores = scores.amax(1).masked_fill(~valid, -float("inf"))
    expected = F.pad(token_scores, (0, 1024 - 1009), value=-float("inf")).reshape(2, 32, 32).amax(-1)
    torch.testing.assert_close(rel, expected, atol=3e-5, rtol=3e-5)
    hit_scores = token_scores.gather(1, anchors.clamp_min(0).long())
    torch.testing.assert_close(hit_scores[torch.isfinite(rel)], rel[torch.isfinite(rel)], atol=3e-5, rtol=3e-5)
    routes, _ = select_question_blocks(q, k, schedule=context_schedule(1009, budget_scaling="sqrt"), query_positions=positions, token_mask=mask, num_heads=2, low_rank_dim=32)
    for head in routes[:, 0]:
        tokens = head[head >= 0]
        assert tokens.numel() == tokens.unique().numel() == routes.shape[-1]
    forced = force_diagnostic_span(routes, (300, 428), 1009)
    assert forced.shape == routes.shape
    for head in forced[:, 0]:
        tokens = head[head >= 0]
        assert tokens.numel() == tokens.unique().numel() == routes.shape[-1]
        assert torch.isin(torch.arange(300, 428, device="cuda"), tokens).all()
    # Max across queries keeps a signal on one query even with other queries
    # pointing in unrelated directions. Reference verifies exact score.
    sq = torch.zeros((1, 2048, 32), device="cuda", dtype=torch.bfloat16)
    sk = torch.zeros_like(sq)
    sq[:, -1, 0], sq[:, -2, 1], sk[:, 900, 0] = 1, 1, 1
    signal, _, _ = question_metadata(sq, sk, [2046, 2047], None, 1, 32, 128)
    assert signal[0, 900 // 32] == 1
    # Integration: raw routing projection capture, GQA expansion, proper
    # rotated attention, causal union, and cleanup against independent torch.
    config = LlamaConfig(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=32, num_hidden_layers=32)
    base = LlamaAttention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    attn = InvestigationAttention(base, query_proxy="raw_question", recent_window=16,
        low_rank_dim=32, budget_scaling="sqrt").eval()
    attn.is_causal = True
    attn.question_span = (985, 993)
    hidden = torch.randn((1, 1009, 128), device="cuda", dtype=torch.bfloat16)
    angles = torch.arange(1009, device="cuda").float()[:, None] * torch.linspace(0.001, 0.05, 32, device="cuda")[None]
    cos, sin = angles.cos()[None].to(torch.bfloat16), angles.sin()[None].to(torch.bfloat16)
    with torch.inference_mode():
        output, _ = attn(hidden, position_embeddings=(cos, sin))
        rq = base.q_proj(hidden).view(1, 1009, 4, 32).transpose(1, 2)
        rk = base.k_proj(hidden).view(1, 1009, 2, 32).transpose(1, 2)
        rv = base.v_proj(hidden).view(1, 1009, 2, 32).transpose(1, 2).repeat_interleave(2, 1).reshape(4, 1009, 32)
        route_q = rq.reshape(4, 1009, 32)
        route_k = rk.repeat_interleave(2, 1).reshape(4, 1009, 32)
        ix, _ = select_question_blocks(route_q, route_k, schedule=context_schedule(1009, budget_scaling="sqrt"), query_positions=list(range(985, 993)) + [1008], recent_window=16, num_heads=4, low_rank_dim=32)
        assert torch.equal(attn.last_selected_indices, ix[:, 0])
        aq, ak = apply_rotary_pos_emb(rq, rk, cos, sin)
        aq = aq.reshape(4, 1009, 32) * base.scaling
        ak = ak.repeat_interleave(2, 1).reshape(4, 1009, 32)
        pos = torch.arange(1009, device="cuda")
        allowed = ((pos[None] <= pos[:, None]) & (pos[None] >= pos[:, None] - 15))[None].expand(4, -1, -1).clone()
        for head in range(4):
            selected = ix[head, 0][ix[head, 0] >= 0].long()
            allowed[head, :, selected] |= selected[None] <= pos[:, None]
        expected_out = ((aq.float() @ ak.float().transpose(-1, -2)).masked_fill(~allowed, -float("inf")).softmax(-1) @ rv.float()).to(torch.bfloat16)
        expected_out = base.o_proj(expected_out.reshape(1, 4, 1009, 32).transpose(1, 2).reshape(1, 1009, 128))
        torch.testing.assert_close(output.float(), expected_out.float(), atol=0.04, rtol=0.04)
    assert attn._raw_q is None and attn._raw_k is None
    assert not base.q_proj._forward_hooks and not base.k_proj._forward_hooks
    # Ordinary selectors must reject any accidentally supplied gold context.
    attn.diagnostic_span = (300, 428)
    try:
        with torch.inference_mode():
            attn(hidden, position_embeddings=(cos, sin))
    except AssertionError:
        pass
    else:
        raise AssertionError("Ordinary selector accepted gold offsets")
    assert attn._raw_q is None and attn._raw_k is None
    print(json.dumps(dict(gpu_checks="PASS", checks=["multiquery_cosine_block_max_reference", "unique_full_budget",
        "oracle_constant_unique_budget", "single_query_signal_survives_max", "raw_projection_and_rotated_attention_union_reference",
        "projection_capture_cleanup", "ordinary_gold_context_rejected"])), flush=True)


if __name__ == "__main__":
    main()
