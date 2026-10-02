"""GPU self-checks for modal_rd.attention before spending on long runs.

1. Kernel + routing: with a vertical budget covering every key, the vs union is
   all causal keys, so output must equal exact SDPA.
2. With a small budget, output must equal an explicit masked softmax over the
   same union (sink + window + routed columns).
3. dense_rd wrapper vs native HF on a real ~16K prompt: identical greedy output.
"""
import torch
import torch.nn.functional as F

from kernels.bigger_bird_flash import bigger_bird_flash


def reference(Q, K, V, sink, window, idx):
    h, n, d = Q.shape
    qp = torch.arange(n, device=Q.device)[:, None]
    kp = torch.arange(n, device=Q.device)[None, :]
    base = (kp <= qp) & ((kp < sink) | (kp > qp - window))
    routed = torch.zeros(h, n, dtype=torch.bool, device=Q.device).scatter_(1, idx.long(), True)
    mask = base[None] | (routed[:, None, :] & (kp <= qp)[None])
    s = (Q.float() @ K.float().transpose(-1, -2)).masked_fill(~mask, -float("inf"))
    return s.softmax(-1) @ V.float()


def main():
    torch.manual_seed(0)
    h, n, d, sink, window = 32, 4096, 128, 64, 512
    Q, K, V = (torch.randn(h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    Q = Q * d ** -0.5
    full = torch.arange(sink, n - window, device="cuda", dtype=torch.int32).expand(h, -1).contiguous()
    out = bigger_bird_flash(Q, K, V, full.view(h, 1, -1), front=sink, window=window, num_heads=h, scale=1.0)
    dense = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True, scale=1.0)[0]
    err1 = (out.float() - dense.float()).abs().max().item()
    idx = torch.stack([torch.randperm(n - window - sink, device="cuda")[:256] + sink for _ in range(h)]).sort(-1).values.int()
    out2 = bigger_bird_flash(Q, K, V, idx.view(h, 1, -1), front=sink, window=window, num_heads=h, scale=1.0)
    err2 = (out2.float() - reference(Q, K, V, sink, window, idx)).abs().max().item()
    print(f"CHECK full-budget vs SDPA max_err={err1:.4g}  small-budget vs masked reference max_err={err2:.4g}", flush=True)
    assert err1 < 2e-2 and err2 < 2e-2

    from transformers import AutoModelForCausalLM
    from modal_rd import attention as rd
    from modal_rd.niah import MODEL_PATH, load_tokenizer, prepare
    tok = load_tokenizer()
    row = prepare(tok, 64000, 1, verify=False)[0]
    ids = tok(row["text"], return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    gen = lambda: tok.decode(model.generate(**ids, max_new_tokens=10, do_sample=False, use_cache=True,
                                            pad_token_id=tok.pad_token_id)[0, row["tokens"]:])
    with torch.inference_mode():
        native = gen()
        rd.install(model, "dense_rd")
        wrapped = gen()
        rd.install(model, "vs", sink=64, window=4096, vertical=2048)
        sparse = gen()
    print(f"CHECK tokens={row['tokens']} answer={row['answer']} native={native!r} dense_rd={wrapped!r} vs={sparse!r}", flush=True)
    assert native == wrapped, "dense_rd wrapper diverges from native HF"
    print("SELFCHECK_PASS", flush=True)


if __name__ == "__main__":
    main()
