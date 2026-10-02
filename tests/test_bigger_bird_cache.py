"""Cache equivalence across a routing-group boundary on a tiny real Llama."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.exp_19_bigger_bird_flash.model_llama import BiggerBirdFlashAttention


@torch.inference_mode()
def main():
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=1024)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    for layer in model.model.layers:
        attn = BiggerBirdFlashAttention(layer.self_attn,
            routing_mode="causal_chunk", route_chunk=128, middle_min=16,
            middle_max=32, window_min=64, window_max=128, local_min=8,
            local_max=16, globals_min=4, globals_max=8, recent_window=17,
            low_rank_dim=64)
        attn.is_causal = True
        layer.self_attn = attn.eval()
    # Explicit additive masks must agree with boolean causal/padding masks.
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 17, 128, device="cuda", dtype=torch.bfloat16)
    allowed = torch.ones(17, 17, device="cuda", dtype=torch.bool).tril()[None, None]
    allowed[..., 3] = False
    additive = torch.zeros_like(allowed, dtype=torch.bfloat16).masked_fill(~allowed, -float("inf"))
    torch.testing.assert_close(attn(hidden, attention_mask=allowed)[0],
                               attn(hidden, attention_mask=additive)[0])
    # Also exercise short-context routing with the default full-prompt mode.
    attn.routing_mode = "last_query"
    assert torch.isfinite(attn(hidden)[0]).all()
    attn.routing_mode = "causal_chunk"
    print("BIGGER_BIRD_ADDITIVE_MASK_AND_SHORT_CONTEXT_PASS", flush=True)
    ids = torch.randint(0, 256, (1, 130), device="cuda")
    # Compare each next-token logit before, at, and after the chunk boundary.
    expected = {}
    for end in range(126, 131):
        expected[end] = model(ids[:, :end], use_cache=False, logits_to_keep=1).logits.clone()
    prefill = model(ids[:, :125], use_cache=True, logits_to_keep=1)
    cache = prefill.past_key_values
    for end in range(126, 131):
        out = model(ids[:, end-1:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
        diff = (out.logits.float() - expected[end].float()).abs().max().item()
        print(f"cache_equivalence_length={end} max_logit_error={diff:.6f}", flush=True)
        torch.testing.assert_close(out.logits.float(), expected[end].float(), atol=0.01, rtol=0.01)
        cache = out.past_key_values
    print("BIGGER_BIRD_CACHE_EQUIVALENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
