"""Matched R&D screen, then conditional full warmed confirmation.

Gold answer offsets are inspected after routing, only for untimed diagnostics.
No selector receives them. Input generation and generation settings match the
previous repository NIAH runs; input byte labels and actual tokens are separate.
"""
import argparse
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from eval.ruler_llama import run_generative as r
from experiments.exp_19_bigger_bird_flash.tokenizer import load_checkpoint_tokenizer
import experiments.exp_19_bigger_bird_flash.model_llama as bb_model
from kernels.bigger_bird_block_routing import select_distinct_blocks, legacy_spans32

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(middle_min=512, middle_max=2048, middle_ratio=64, window_min=1024,
    window_max=8192, local_min=128, local_max=512, globals_min=32, globals_max=64,
    recent_window=128, diversity=0.05, low_rank_dim=128, routing_mode="last_query",
    route_chunk=1024, use_triton=True, block_m=64, block_n=64,
    budget_scaling="sqrt", budget_factor=1.0)
VARIANTS = {
    "dense_flash": (0, {}, None),
    "sqrt_spans32": (19, {}, "legacy"),
    "block_distinct": (19, {}, None),
    "block_coverage": (19, {}, 0.125),
    "block_coverage_1p5x": (19, {"budget_factor": 1.5}, 0.125),
    "block_prefix4": (19, {"routing_mode": "bounded_prefix", "max_route_groups": 4}, 0.125),
}


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def source_hashes():
    files = ["kernels/bigger_bird_block_routing.py", "kernels/bigger_bird_routing.py",
             "kernels/bigger_bird_flash.py", "experiments/exp_19_bigger_bird_flash/model_llama.py",
             "eval/ruler_llama/run_generative.py", "scripts/run_block_routing_rd.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def prepare(tokenizer, seq_bytes, count):
    data = r.build_ruler_dataset(task="niah", seq_len=seq_bytes, needle_depth=0.5,
        train_samples=10, eval_samples=128, seed=42)["validation"]
    rows = []
    for idx in range(count):
        text = r._ids_to_text(data[idx]["input_ids"]).rstrip() + " The answer is:"
        question = re.search(r"What is the special magic ([\w-]+)\?", text)
        assert question, (seq_bytes, idx)
        needle = re.search(r"The special magic " + re.escape(question[1]) + r" is: (\d+)\.", text)
        assert needle, (seq_bytes, idx)
        encoded = tokenizer(text, return_offsets_mapping=True, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        # Refuse silently truncated contexts or prompts that exceed the model
        # limit during uncached ten-token generation.
        assert len(ids) + 10 <= 131072, (seq_bytes, idx, len(ids))
        assert seq_bytes * 0.225 <= len(ids) <= seq_bytes * 0.275, (seq_bytes, idx, len(ids))
        spans = {}
        for label, start, end in (("needle", needle.start(), needle.end()),
                                  ("number", needle.start(1), needle.end(1))):
            hits = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]
            assert hits, (idx, label)
            spans[label] = [min(hits), max(hits) + 1]
        assert int(needle[1][-1]) == int(data[idx]["labels"])
        rows.append(dict(idx=idx, text=text, label=int(data[idx]["labels"]), answer=needle[1],
            tokens=len(ids), token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            source_sha256=hashlib.sha256(text.encode()).hexdigest(), spans=spans))
    return rows


def needle_diagnostics(model, row):
    """Post-selection observations, excluded from timed passes."""
    layers = model.model.layers
    vectors = []
    for layer in layers:
        indices = layer.self_attn.last_selected_indices
        valid = indices >= 0
        ordered = indices.sort(-1).values
        unique = (ordered >= 0)
        unique[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
        begin, end = row["spans"]["number"]
        hits = ((indices >= begin) & (indices < end)).sum(-1)
        nb, ne = row["spans"]["needle"]
        context = ((indices >= nb) & (indices < ne)).sum(-1)
        vectors.append(torch.stack((valid.sum(-1).float().mean(), unique.sum(-1).float().mean(),
            valid.sum(-1).float().min(), (hits > 0).float().mean(),
            (hits >= end - begin).float().mean(), (hits > 0).any().float(),
            (hits >= end - begin).any().float(), context.float().mean() / (ne - nb),
            indices.new_tensor(indices.shape[-1]).float())))
    return torch.stack(vectors)


def generate_one(model, tokenizer, row, exp, diagnostics=False):
    records, spans = [], []
    def before(_mod, args, kwargs):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        records.append([start, None, kwargs["input_ids"].shape[1]])
    def after(_mod, args, kwargs, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        records[-1][1] = end
        if diagnostics and exp == 19 and len(records) == 1:
            spans.append(needle_diagnostics(model, row))
    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=131072).to("cuda")
        assert inputs["input_ids"].shape[1] == row["tokens"]
        # The first result for each method verifies the exact model input too.
        # Full manifests were built once from the identical tokenizer above.
        with torch.inference_mode(), (sdpa_kernel(SDPBackend.FLASH_ATTENTION) if exp == 0 else contextlib.nullcontext()):
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        new_ids = output[0, row["tokens"]:].tolist()
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    finally:
        pre.remove()
        post.remove()
    digit = r.parse_prediction(generated, task="niah")
    first = re.search(r"\d+", generated)
    result = dict(idx=row["idx"], generated=generated, generated_ids=new_ids,
        generated_token_count=len(new_ids), forward_count=len(records),
        last_digit_correct=digit == row["label"], exact_correct=bool(first and first[0] == row["answer"]),
        time_seconds=seconds, prefill_seconds=records[0][0].elapsed_time(records[0][1]) / 1000,
        forward_seconds=[a.elapsed_time(b) / 1000 for a, b, _ in records],
        forward_context_tokens=[n for _, _, n in records], token_sha256=row["token_sha256"])
    if spans:
        keys = ["valid_tokens_mean", "unique_tokens_mean", "valid_tokens_min",
                "number_any_head_fraction", "number_full_head_fraction", "number_any_in_any_head",
                "number_full_in_any_head", "needle_statement_coverage_mean", "allocated_slots"]
        result["routing_by_layer"] = [dict(layer=i, **dict(zip(keys, vec))) for i, vec in enumerate(spans[0].cpu().tolist())]
    return result


def run_variant(name, seq_bytes, rows, output, stage, hashes):
    exp, overrides, coverage = VARIANTS[name]
    registry = r.EXP_REGISTRY[19]
    original_selector = bb_model.select_bigger_bird
    kwargs = {**BASE, **overrides} if exp else {}
    print(f"PHASE_START stage={stage} variant={name} bytes={seq_bytes} examples={len(rows)}", flush=True)
    if exp:
        r.EXP_REGISTRY[19] = (registry[0], registry[1], kwargs)
        bb_model.select_bigger_bird = legacy_spans32(original_selector) if coverage == "legacy" else functools.partial(select_distinct_blocks, coverage_fraction=coverage)
    model = None
    try:
        started = time.perf_counter()
        model = r.build_generative_model(exp).to("cuda")
        assert next(model.parameters()).dtype == torch.bfloat16
        assert "3g.40gb" in torch.cuda.get_device_name(0).lower()
        setup_seconds = time.perf_counter() - started
        warm = []
        for row in rows:
            warm.append(generate_one(model, tokenizer, row, exp, diagnostics=True))
            print(f"WARM variant={name} idx={row['idx']} exact={warm[-1]['exact_correct']} generated={warm[-1]['generated']!r}", flush=True)
            write_json(output / f"{name}_bytes{seq_bytes}_warm_progress.json", warm)
        print(f"WARM_COMPLETE variant={name} examples={len(warm)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        timed = []
        for row in rows:
            timed.append(generate_one(model, tokenizer, row, exp))
            print(f"TIMED variant={name} idx={row['idx']} exact={timed[-1]['exact_correct']} seconds={timed[-1]['time_seconds']:.3f}", flush=True)
            write_json(output / f"{name}_bytes{seq_bytes}_timed_progress.json", timed)
        assert all(a["generated_ids"] == b["generated_ids"] for a, b in zip(warm, timed)), "warm/timed deterministic predictions differ"
        result = dict(stage=stage, name=name, exp=exp, seq_len_bytes=seq_bytes,
            actual_tokens_min=min(x["tokens"] for x in rows), actual_tokens_max=max(x["tokens"] for x in rows),
            n_examples=len(rows), exact_number_correct=sum(x["exact_correct"] for x in timed),
            last_digit_correct=sum(x["last_digit_correct"] for x in timed),
            held_out_exact_correct=sum(x["exact_correct"] for x in timed if x["idx"] >= 6),
            held_out_examples=sum(x["idx"] >= 6 for x in timed),
            time_seconds=sum(x["time_seconds"] for x in timed),
            prefill_seconds=sum(x["prefill_seconds"] for x in timed),
            peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
            setup_seconds=setup_seconds, warmup_examples=len(warm),
            protocol="all_prompts_warm_then_same_prompts_timed", depth=0.5, seed=42, eval_samples=128,
            max_new_tokens=10, use_cache=False, dtype="bfloat16", greedy=True,
            model="DeepSeek-R1-Distill-Llama-8B", gpu=torch.cuda.get_device_name(0),
            job_id=os.environ["SLURM_JOB_ID"], git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            attention_config=kwargs, selector_config=dict(block_size=32, coverage_fraction=coverage,
                implementation="legacy_token_anchors" if coverage == "legacy" else "distinct_block_max_mmr"),
            sources=hashes, examples=timed, diagnostic_examples=warm,
            timing_notes="Wall time includes tokenization, transfer and greedy no-cache generation. Prefill CUDA events measure the first full-context model forward, independently of generated length. Diagnostics run only in the untimed warm pass.",
            causality_notes="Attention edges remain causal. last_query routing depends on later prompt queries. bounded_prefix uses only the visible prefix for each group; layout depends on total length.")
        write_json(output / f"{name}_bytes{seq_bytes}.json", result)
        print(f"PHASE_COMPLETE variant={name} bytes={seq_bytes} exact={result['exact_number_correct']}/{len(rows)} seconds={result['time_seconds']:.3f} prefill={result['prefill_seconds']:.3f} mem_gb={result['peak_memory_gb']:.3f}", flush=True)
        return result
    finally:
        r.EXP_REGISTRY[19] = registry
        bb_model.select_bigger_bird = original_selector
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["screen", "confirm"])
    parser.add_argument("--screen-job")
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / "research_outputs" / f"{args.stage}_{job}"
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    write_json(output / "source_hashes.json", hashes)
    tokenizer = load_checkpoint_tokenizer(r.MODEL_PATH)
    tokenizer.padding_side = "left"
    if args.stage == "screen":
        rows = prepare(tokenizer, 512000, 6)
        write_json(output / "prompt_manifest_bytes512000.json", [{k: v for k, v in row.items() if k != "text"} for row in rows])
        results = [run_variant(name, 512000, rows, output, "screen", hashes) for name in VARIANTS]
        dense, legacy = results[:2]
        eligible = [x for x in results[2:] if x["exact_number_correct"] >= 4 and x["exact_number_correct"] > legacy["exact_number_correct"] and x["prefill_seconds"] < dense["prefill_seconds"]]
        eligible.sort(key=lambda x: (-x["exact_number_correct"], x["prefill_seconds"]))
        write_json(output / "screen_summary.json", dict(results=[{k: v for k, v in x.items() if k not in ("examples", "diagnostic_examples")} for x in results],
            strongest=eligible[0]["name"] if eligible else None,
            selection_rule="At least 4/6 exact, better than legacy span control, faster first-forward prefill than matched dense; rank exact accuracy then prefill time.",
            tuning_indices=list(range(6)), confirmation_held_out_indices=list(range(6, 30))))
        print("SCREEN_COMPLETE", flush=True)
    else:
        summary = json.loads((ROOT / "research_outputs" / f"screen_{args.screen_job}" / "screen_summary.json").read_text())
        winner = summary["strongest"]
        if winner is None:
            write_json(output / "confirmation_skipped.json", {"reason": "No new selector met the predeclared quality and speed screen", "screen_job": args.screen_job})
            print("CONFIRMATION_SKIPPED_NO_QUALIFYING_SELECTOR", flush=True)
        else:
            for seq in (261000, 512000):
                rows = prepare(tokenizer, seq, 30)
                write_json(output / f"prompt_manifest_bytes{seq}.json", [{k: v for k, v in row.items() if k != "text"} for row in rows])
                for name in ("dense_flash", winner):
                    run_variant(name, seq, rows, output, "confirm", hashes)
            write_json(output / "confirmation_complete.json", {"winner": winner, "screen_job": args.screen_job, "contexts": [261000, 512000]})
            print("CONFIRMATION_COMPLETE", flush=True)
