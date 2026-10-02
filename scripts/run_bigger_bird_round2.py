"""Round2 diagnostic and question-route screen; conditional independent confirmation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import torch
from scripts import run_block_routing_rd as h

ROOT = h.ROOT
ARMS = {
    "dense_flash": {},
    "coverage_control": {"query_proxy": "coverage"},
    "oracle_all_layers": {"query_proxy": "coverage", "diagnostic_oracle": "all"},
    "oracle_last8_layers": {"query_proxy": "coverage", "diagnostic_oracle": "late8"},
    "rotated_question_max": {"query_proxy": "rotated_question"},
    "raw_question_max": {"query_proxy": "raw_question"},
    "raw_question_max_local512": {"query_proxy": "raw_question", "recent_window": 512},
}


def source_hashes():
    files = list(h.source_hashes()) + ["kernels/bigger_bird_query_routing.py",
        "experiments/exp_19_bigger_bird_flash/model_investigation.py",
        "scripts/run_bigger_bird_round2.py", "scripts/test_bigger_bird_round2.py",
        "scripts/validate_bigger_bird_round2.sbatch", "scripts/screen_bigger_bird_round2.sbatch",
        "scripts/confirm_bigger_bird_round2.sbatch"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}


def prepare(tokenizer, seq_bytes, count):
    rows = h.prepare(tokenizer, seq_bytes, count)
    for row in rows:
        # These positions are obtained from the visible question suffix, with
        # no reference to the answer statement or its value.
        question = re.search(r"What is the special magic [\w-]+\?", row["text"])
        offsets = tokenizer(row["text"], return_offsets_mapping=True, truncation=False)["offset_mapping"]
        ids = [i for i, (a, b) in enumerate(offsets) if b > question.start() and a < question.end() and b > a]
        row["question_tokens"] = [min(ids), max(ids) + 1]
        assert row["question_tokens"][0] > row["tokens"] - 128
    return rows


def set_context(model, row, oracle):
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.question_span = tuple(row["question_tokens"])
        attn.diagnostic_span = None
        if oracle:
            start, end = row["spans"]["needle"]
            # 128 tokens centered on the statement. Constant budget; matched
            # coverage route slots are replaced, rather than added.
            begin = max(0, min(row["tokens"] - 128, (start + end) // 2 - 64))
            attn.diagnostic_span = (begin, begin + 128)


def run_arm(name, seq_bytes, rows, tokenizer, output, stage, hashes):
    oracle = name.startswith("oracle_")
    exp = 0 if name == "dense_flash" else 19
    registry = h.r.EXP_REGISTRY[19]
    kwargs = {**h.BASE, **ARMS[name]} if exp else {}
    if exp:
        h.r.EXP_REGISTRY[19] = ("experiments.exp_19_bigger_bird_flash.model_investigation", "InvestigationAttention", kwargs)
    model = None
    print(f"PHASE_START stage={stage} variant={name} bytes={seq_bytes} examples={len(rows)} diagnostic_only={oracle}", flush=True)
    try:
        started = time.perf_counter()
        model = h.r.build_generative_model(exp).to("cuda")
        assert next(model.parameters()).dtype == torch.bfloat16
        assert "3g.40gb" in torch.cuda.get_device_name(0).lower()
        setup_seconds = time.perf_counter() - started
        warm, timed = [], []
        for row in rows:
            if exp:
                set_context(model, row, oracle)
            warm.append(h.generate_one(model, tokenizer, row, exp, diagnostics=True))
            print(f"WARM variant={name} idx={row['idx']} exact={warm[-1]['exact_correct']} generated={warm[-1]['generated']!r}", flush=True)
            h.write_json(output / f"{name}_bytes{seq_bytes}_warm_progress.json", warm)
        print(f"WARM_COMPLETE variant={name} examples={len(warm)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            if exp:
                set_context(model, row, oracle)
            timed.append(h.generate_one(model, tokenizer, row, exp))
            print(f"TIMED variant={name} idx={row['idx']} exact={timed[-1]['exact_correct']} seconds={timed[-1]['time_seconds']:.3f}", flush=True)
            h.write_json(output / f"{name}_bytes{seq_bytes}_timed_progress.json", timed)
        assert all(a["generated_ids"] == b["generated_ids"] for a, b in zip(warm, timed))
        result = dict(stage=stage, name=name, exp=exp, diagnostic_only=oracle,
            seq_len_bytes=seq_bytes, actual_tokens_min=min(x["tokens"] for x in rows), actual_tokens_max=max(x["tokens"] for x in rows),
            n_examples=len(rows), exact_number_correct=sum(x["exact_correct"] for x in timed), last_digit_correct=sum(x["last_digit_correct"] for x in timed),
            held_out_exact_correct=sum(x["exact_correct"] for x in timed if x["idx"] >= 6), held_out_examples=sum(x["idx"] >= 6 for x in timed),
            time_seconds=sum(x["time_seconds"] for x in timed), prefill_seconds=sum(x["prefill_seconds"] for x in timed),
            peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9, setup_seconds=setup_seconds, warmup_examples=len(warm),
            protocol="all_prompts_warm_then_same_prompts_timed", depth=0.5, seed=42, eval_samples=128,
            max_new_tokens=10, use_cache=False, dtype="bfloat16", greedy=True, model="DeepSeek-R1-Distill-Llama-8B",
            gpu=torch.cuda.get_device_name(0), job_id=os.environ["SLURM_JOB_ID"], git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            attention_config=kwargs, selector_config=dict(block_size=32, coverage_fraction=0.125, query_positions="up_to8_question_tokens_plus_latest", score="maximum_cosine" if "question" in name else "last_query_dot", diagnostic_only=oracle),
            sources=hashes, examples=timed, diagnostic_examples=warm,
            timing_notes="Wall time includes tokenization, transfer and variable-length greedy generation. Prefill CUDA events measure the first model forward. Routing diagnostics only in untimed warm pass.",
            causality_notes="Attention edges causal; global question-aware selection uses later prompt queries for earlier tokens. No prefix-invariance or KV-cache equivalence claim.",
            diagnostic_notes="Oracle arms forcibly retain 128 tokens centered on the gold statement in all heads of the selected layers; answer-aware and never benchmark eligible." if oracle else "Gold spans are inspected only after route selection in untimed diagnostics, and are never supplied to this selector.")
        h.write_json(output / f"{name}_bytes{seq_bytes}.json", result)
        print(f"PHASE_COMPLETE variant={name} exact={result['exact_number_correct']}/{len(rows)} prefill={result['prefill_seconds']:.3f} wall={result['time_seconds']:.3f} memory={result['peak_memory_gb']:.3f}", flush=True)
        assert hashes == source_hashes(), "Sources changed during run"
        return result
    finally:
        h.r.EXP_REGISTRY[19] = registry
        if model is not None:
            for layer in model.model.layers:
                layer.self_attn.question_span = layer.self_attn.diagnostic_span = None
            del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "confirm"))
    parser.add_argument("--screen-job")
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / "research_outputs" / f"round2_{args.stage}_{job}"
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    h.write_json(output / "source_hashes.json", hashes)
    tokenizer = h.load_checkpoint_tokenizer(h.r.MODEL_PATH)
    tokenizer.padding_side = "left"
    if args.stage == "screen":
        rows = prepare(tokenizer, 512000, 6)
        h.write_json(output / "prompt_manifest_bytes512000.json", [{k: v for k, v in x.items() if k != "text"} for x in rows])
        results = [run_arm(name, 512000, rows[:3] if name.startswith("oracle_") else rows, tokenizer, output, "screen", hashes) for name in ARMS]
        dense, control = results[:2]
        eligible = [x for x in results[2:] if not x["diagnostic_only"] and x["exact_number_correct"] >= 4 and x["exact_number_correct"] > control["exact_number_correct"] and x["prefill_seconds"] < dense["prefill_seconds"]]
        eligible.sort(key=lambda x: (-x["exact_number_correct"], x["prefill_seconds"]))
        h.write_json(output / "screen_summary.json", dict(results=[{k: v for k, v in x.items() if k not in ("examples", "diagnostic_examples")} for x in results],
            strongest=eligible[0]["name"] if eligible else None,
            selection_rule="Only ordinary selectors: >=4/6 exact, better than matched coverage control, faster mean first-forward prefill than dense; rank exact then prefill. Oracles are diagnostics excluded from selection.",
            tuning_indices=list(range(6)), oracle_indices=list(range(3)), confirmation_held_out_indices=list(range(6, 30))))
        print("SCREEN_COMPLETE", flush=True)
    else:
        summary = json.loads((ROOT / "research_outputs" / f"round2_screen_{args.screen_job}" / "screen_summary.json").read_text())
        assert all(x["sources"] == hashes for x in summary["results"])
        winner = summary["strongest"]
        if winner is None:
            h.write_json(output / "confirmation_skipped.json", dict(reason="No ordinary selector passed predeclared screen", screen_job=args.screen_job))
            print("CONFIRMATION_SKIPPED_NO_QUALIFYING_SELECTOR", flush=True)
            return
        assert winner in ARMS and not winner.startswith("oracle_")
        for seq in (261000, 512000):
            rows = prepare(tokenizer, seq, 30)
            h.write_json(output / f"prompt_manifest_bytes{seq}.json", [{k: v for k, v in x.items() if k != "text"} for x in rows])
            for name in ("dense_flash", winner):
                run_arm(name, seq, rows, tokenizer, output, "confirm", hashes)
        h.write_json(output / "confirmation_complete.json", dict(winner=winner, screen_job=args.screen_job, contexts=[261000, 512000]))
        print("CONFIRMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
