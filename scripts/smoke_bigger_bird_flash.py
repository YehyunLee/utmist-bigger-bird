"""Paired full-model exact-number retrieval with actual tokenizer lengths.

This is a custom RULER-style smoke test, not the official RULER suite. Native
HF dense FlashAttention and Bigger Bird use the same prompts and cache policy.
Compile/model loading are excluded; all variants produce the same fixed number
of output tokens so stopping behavior cannot manufacture a latency win.
"""
import argparse
import gc
import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel, SDPBackend
from experiments.exp_19_bigger_bird_flash.model_llama import BiggerBirdFlashAttention
from experiments.exp_19_bigger_bird_flash.tokenizer import load_checkpoint_tokenizer


def prompt_ids(tokenizer, n, depth, gold):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    header = encode("Read the following text carefully. Remember the special number.\n")
    needle = encode(f"\nThe special number is {gold}. Remember it.\n")
    question = encode("\nWhat is the special number? Answer with only the number. The answer is:")
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    filler = encode("The grass is green. The sky is blue. The sun is yellow. There and back again. ")
    remaining = n - len(bos) - len(header) - len(needle) - len(question)
    before = int(remaining * depth)
    noise = (filler * ((remaining + len(filler) - 1)//len(filler)))[:remaining]
    start = len(bos) + len(header) + before
    ids = bos + header + noise[:before] + needle + noise[before:] + question
    assert len(ids) == n
    return torch.tensor([ids], device="cuda"), (start, start + len(needle))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="2048,4096,8192")
    p.add_argument("--depths", default="0.25,0.75")
    p.add_argument("--new-tokens", type=int, default=6)
    p.add_argument("--modes", default="last_query,causal_chunk")
    p.add_argument("--cached-dense", action="store_true", help="Also measure native dense generation with a KV cache")
    p.add_argument("--cached-sparse", action="store_true", help="Also measure Bigger Bird with a full GQA KV cache")
    p.add_argument("--only-cached", action="store_true", help="Skip uncached variants")
    p.add_argument("--block-m", type=int, default=64)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--route-chunk", type=int, default=1024)
    p.add_argument("--model", default="/scratch/thomas7/models/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--output", default="benchmarks/bigger_bird_flash/smoke.json")
    args = p.parse_args()
    tokenizer = load_checkpoint_tokenizer(args.model)
    print("Loading frozen R1-Llama-8B with native SDPA dense attention", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, low_cpu_mem_usage=True).cuda().eval()
    layers = model.model.layers
    native = [layer.self_attn for layer in layers]
    variants = {"dense_flash": native}
    if args.cached_dense:
        variants["dense_flash_cached"] = native
    for mode in args.modes.split(","):
        modules = [BiggerBirdFlashAttention(attn, routing_mode=mode,
            block_m=args.block_m, block_n=args.block_n, route_chunk=args.route_chunk) for attn in native]
        for module in modules:
            module.is_causal = True
            module.eval()
        variants["bigger_bird_" + mode] = modules
        if args.cached_sparse:
            variants["bigger_bird_" + mode + "_cached"] = modules
    results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    def save():
        with open(args.output, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                cache_policy="per_result", generated_tokens=args.new_tokens,
                benchmark="custom_exact_number_retrieval_smoke", results=results), f, indent=2)
    for n in map(int, args.seqs.split(",")):
        for variant, modules in variants.items():
            if args.only_cached and not variant.endswith("_cached"):
                continue
            use_cache = variant.endswith("_cached")
            for layer, attn in zip(layers, modules):
                layer.self_attn = attn
            warm, _ = prompt_ids(tokenizer, n, 0.5, "7654321")
            # Warm every generated length too: Triton specializes on N, so a
            # single prefill would leave N+1..N+T compilation in the timing.
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    model.generate(warm, use_cache=use_cache, do_sample=False,
                        max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                        pad_token_id=tokenizer.eos_token_id)
            except torch.OutOfMemoryError:
                item = dict(variant=variant, input_tokens=n, cache=use_cache,
                    status="oom_during_warmup", timing_valid=False)
                print(json.dumps(item), flush=True)
                results.append(item)
                save()
                del warm
                gc.collect()
                torch.cuda.empty_cache()
                continue
            del warm
            torch.cuda.synchronize()
            for sample, depth in enumerate(map(float, args.depths.split(","))):
                gold = ("1234567", "8912345", "5647382")[sample % 3]
                ids, span = prompt_ids(tokenizer, n, depth, gold)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = model.generate(ids, use_cache=use_cache, do_sample=False,
                        max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                        pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                raw_generated = tokenizer.decode(out[0, n:], skip_special_tokens=True)
                generated = raw_generated
                item = dict(variant=variant, input_tokens=n, depth=depth, gold=gold,
                    generated=generated, exact_number_correct=gold in re.sub(r"\s", "", generated),
                    seconds=elapsed, peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                    output_tokens=int(out.shape[-1]-n), cache=use_cache,
                    raw_generated=raw_generated)
                if variant.startswith("bigger_bird_"):
                    stats = []
                    for attn in modules:
                        idx = attn.last_selected_indices
                        hit = (idx >= span[0]) & (idx < span[1])
                        stats.append(dict(head_fraction_with_needle=hit.any(-1).float().mean().item(),
                            mean_unique_routes=(idx >= 0).sum(-1).float().mean().item()))
                    item["selection"] = dict(modules[-1].last_diagnostics,
                        mean_head_fraction_with_needle=sum(s["head_fraction_with_needle"] for s in stats)/len(stats),
                        mean_unique_routes=sum(s["mean_unique_routes"] for s in stats)/len(stats))
                print(json.dumps(item), flush=True)
                results.append(item)
                save()
                del ids, out
            gc.collect()
            torch.cuda.empty_cache()
    save()
    print("BIGGER_BIRD_MODEL_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
