"""Forced dense FlashAttention vs sparse kernel AND complete routing+attention."""
import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from kernels.bigger_bird_flash import bigger_bird_flash
from kernels.bigger_bird_routing import context_schedule, select_bigger_bird


def timing(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="2048,4096,8192,16384,32768")
    p.add_argument("--output", default="benchmarks/bigger_bird_flash/microbench.json")
    p.add_argument("--block-m", type=int, default=64)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--mode", default="last_query", choices=("last_query", "causal_chunk"))
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    torch.manual_seed(42)
    results = []
    for n in map(int, args.seqs.split(",")):
        q4 = torch.randn(1, 32, n, 128, device="cuda", dtype=torch.bfloat16)
        k4, v4 = torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16), torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16)
        q = q4.reshape(32, n, 128) / 128**0.5
        k, v = k4.repeat_interleave(4, 1).reshape(32, n, 128), v4.repeat_interleave(4, 1).reshape(32, n, 128)
        schedule = context_schedule(n)
        def route():
            return select_bigger_bird(q, k, schedule=schedule, num_heads=32, routing_mode=args.mode)
        idx, chunk = route()
        def sparse():
            return bigger_bird_flash(q, k, v, idx, front=0, window=128,
                                    route_chunk=chunk, block_m=args.block_m, block_n=args.block_n)
        def complete():
            selected, group = route()
            return bigger_bird_flash(q, k, v, selected, front=0, window=128,
                                    route_chunk=group, block_m=args.block_m, block_n=args.block_n)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            dense_ms = timing(lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, enable_gqa=True), iters=args.iters)
        route_ms = timing(route, iters=args.iters)
        sparse_ms = timing(sparse, iters=args.iters)
        complete_ms = timing(complete, iters=args.iters)
        item = dict(n=n, gpu=torch.cuda.get_device_name(0), dtype="bfloat16", heads=32, kv_heads=8,
            schedule=schedule, mode=args.mode, block_m=args.block_m, block_n=args.block_n,
            dense_backend="forced_flash_attention", dense_ms=dense_ms,
            routing_ms=route_ms, sparse_attention_ms=sparse_ms, sparse_complete_ms=complete_ms,
            attention_speedup=dense_ms/sparse_ms, complete_speedup=dense_ms/complete_ms,
            mean_unique_routes=(idx >= 0).sum(-1).float().mean().item(),
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        print(json.dumps(item), flush=True)
        results.append(item)
        del q4, k4, v4, q, k, v, idx
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), torch=torch.__version__, results=results), f, indent=2)


if __name__ == "__main__":
    main()
