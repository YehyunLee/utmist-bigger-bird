"""Teacher-forced diagnostic separating route choice from layerwise sparse effects.

At the exact round2 127K-token prompts, score the known number continuation
under dense, ordinary sparse, answer-aware sparse, and dense/sparse layer
hybrids.  This is a mechanistic diagnostic, not a benchmark or deployable
configuration. No generation-time or KV-cache conclusion is drawn.
"""
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts import run_block_routing_rd as h
from scripts import run_bigger_bird_round2 as r2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_outputs" / f"round3_probe_{os.environ['SLURM_JOB_ID']}"
BASE = {**h.BASE, "query_proxy": "coverage"}
N_LAYERS = 32
CONDITIONS = [
    ("dense_flash", "dense", tuple()),
    ("dense_wrapper", "oracle", tuple(range(N_LAYERS))),
    ("coverage_control", "sparse", tuple()),
    ("oracle_sparse_all", "oracle", tuple()),
    ("dense_first8_oracle", "oracle", tuple(range(0, 8))),
    ("dense_last8_oracle", "oracle", tuple(range(24, 32))),
    ("dense_first16_oracle", "oracle", tuple(range(0, 16))),
    ("dense_last16_oracle", "oracle", tuple(range(16, 32))),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    files = list(r2.source_hashes()) + [
        "experiments/exp_19_bigger_bird_flash/model_round3_probe.py",
        "scripts/run_round3_probe.py",
        "scripts/probe_bigger_bird_round3.sbatch",
    ]
    return {name: sha(ROOT / name) for name in sorted(set(files))}


def attention_hook(position_id, bucket, idx):
    def hook(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        bucket[idx] = value[0, position_id, :].detach().float().cpu()
    return hook


def run_condition(name, mode, dense_layers, rows, tokenizer, expected_sources):
    exp = 0 if mode == "dense" else 19
    old = h.r.EXP_REGISTRY[19]
    kwargs = {}
    if exp:
        oracle = mode == "oracle"
        kwargs = {
            **BASE,
            "query_proxy": "coverage",
            "diagnostic_oracle": "all" if oracle else "none",
            "dense_layers": list(dense_layers),
        }
        h.r.EXP_REGISTRY[19] = (
            "experiments.exp_19_bigger_bird_flash.model_round3_probe",
            "Round3ProbeAttention",
            kwargs,
        )
    model = None
    outputs = []
    try:
        started = time.perf_counter()
        model = h.r.build_generative_model(exp).to("cuda").eval()
        setup = time.perf_counter() - started
        assert next(model.parameters()).dtype == torch.bfloat16
        assert "3g.40gb" in torch.cuda.get_device_name(0).lower()
        for row in rows:
            prefix = tokenizer(row["text"], return_tensors="pt", truncation=False)
            prefix_ids = prefix["input_ids"][0].tolist()
            full_text = row["text"] + " " + row["answer"]
            encoded = tokenizer(full_text, return_tensors="pt", truncation=False)
            ids = encoded["input_ids"][0].tolist()
            n = len(prefix_ids)
            assert ids[:n] == prefix_ids, f"Tokenizer boundary changed for prompt {row['idx']}"
            target = ids[n:]
            assert target and n + len(target) <= 131072
            query_positions = [n - 1]
            if exp:
                r2.set_context(model, row, mode == "oracle")
            captured = {}
            hooks = [
                layer.self_attn.register_forward_hook(attention_hook(query_positions[0], captured, i))
                for i, layer in enumerate(model.model.layers)
            ]
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if exp == 0 else nullcontext()
                with torch.inference_mode(), ctx:
                    # Calling the backbone avoids materializing [sequence,
                    # vocabulary] logits. Project only the few answer-query
                    # states needed for teacher-forced scoring.
                    result = model.model(**encoded.to("cuda"), use_cache=False, return_dict=True)
                torch.cuda.synchronize()
            finally:
                for hook in hooks:
                    hook.remove()
            hidden = result.last_hidden_state[:, n - 1:n + len(target) - 1, :]
            # `hidden` was created inside inference_mode. Project and score it
            # inside inference_mode too; leaving the context makes autograd
            # reject inference tensors even though this probe never backprops.
            with torch.inference_mode():
                logits = model.lm_head(hidden)[0].float()
                target_t = torch.tensor(target, device=logits.device)
                logp = F.log_softmax(logits, dim=-1).gather(-1, target_t[:, None]).squeeze(-1)
                top = logits.argmax(-1)
            layer_traces = []
            for i, layer in enumerate(model.model.layers):
                selected = getattr(layer.self_attn, "last_selected_indices", None)
                if selected is None:
                    retained = 1.0
                    route_kind = "dense"
                else:
                    start, end = row["spans"]["number"]
                    retained = float(((selected >= start) & (selected < end)).sum(-1).eq(end - start).float().mean().item())
                    route_kind = "sparse"
                layer_traces.append({"layer": i, "route_kind": route_kind,
                                     "gold_number_full_in_head_fraction": retained})
            outputs.append({
                "idx": row["idx"], "answer": row["answer"],
                "prompt_tokens": n, "teacher_forced_answer_tokens": len(target),
                "token_sha256": row["token_sha256"], "source_sha256": row["source_sha256"],
                "gold_answer_mean_nll": float((-logp.mean()).item()),
                "gold_answer_total_nll": float((-logp.sum()).item()),
                "gold_token_top1_count": int((top == target_t).sum().item()),
                "gold_token_top1_total": len(target),
                "all_gold_tokens_top1": bool(torch.equal(top, target_t)),
                "first_target_top1": bool(top[0].item() == target[0]),
                "first_target_logprob": float(logp[0].item()),
                "top1_continuation": tokenizer.decode(top.tolist(), skip_special_tokens=True),
                "target_continuation": tokenizer.decode(target, skip_special_tokens=True),
                "attention_output_at_answer_queries_by_layer": {
                    str(i): captured[i].tolist() for i in sorted(captured)
                },
                "route_by_layer": layer_traces,
                "forward_seconds": float(time.perf_counter() - started),
            })
            del logits, hidden, result
            torch.cuda.empty_cache()
        result = {
            "name": name, "mode": mode, "dense_layers": list(dense_layers),
            "stage": "teacher_forced_layer_diagnostic", "seq_len_bytes": 512000,
            "actual_tokens_min": min(r["tokens"] for r in rows),
            "actual_tokens_max": max(r["tokens"] for r in rows),
            "n_examples": len(rows), "setup_seconds": setup,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "attention_config": kwargs, "examples": outputs, "sources": expected_sources,
            "protocol": "single_uncached_teacher_forced_forward_per_prompt",
            "note": "Diagnostic only. Gold continuation and, for oracle modes, gold statement span are used; not a deployable benchmark.",
        }
        h.write_json(OUT / f"{name}_bytes512000_teacher_forced.json", result)
        print(f"PHASE_COMPLETE name={name} target_nll={sum(x['gold_answer_total_nll'] for x in outputs):.4f} "
              f"top1={sum(x['gold_token_top1_count'] for x in outputs)}/"
              f"{sum(x['gold_token_top1_total'] for x in outputs)} "
              f"prefill_wall={sum(x['forward_seconds'] for x in outputs):.3f} "
              f"memory_gb={result['peak_memory_gb']:.3f}", flush=True)
        return result
    finally:
        h.r.EXP_REGISTRY[19] = old
        if model is not None:
            for layer in model.model.layers:
                if hasattr(layer.self_attn, "question_span"):
                    layer.self_attn.question_span = None
                    layer.self_attn.diagnostic_span = None
            del model
        torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    h.write_json(OUT / "source_hashes.json", hashes)
    tokenizer = h.load_checkpoint_tokenizer(h.r.MODEL_PATH)
    tokenizer.padding_side = "left"
    rows = r2.prepare(tokenizer, 512000, 3)
    h.write_json(OUT / "prompt_manifest_bytes512000.json",
                 [{k: v for k, v in row.items() if k != "text"} for row in rows])
    all_results = []
    for name, mode, dense_layers in CONDITIONS:
        print(f"PHASE_START name={name} mode={mode} dense_layers={list(dense_layers)}", flush=True)
        all_results.append(run_condition(name, mode, dense_layers, rows, tokenizer, hashes))
        assert source_hashes() == hashes, "Round3 sources changed during the job"
        h.write_json(OUT / "probe_progress.json", [
            {k: v for k, v in x.items() if k != "examples"} for x in all_results
        ])
    baseline = all_results[0]
    baseline_rows = {x["idx"]: x for x in baseline["examples"]}
    summary = []
    for item in all_results:
        prompt_rows = []
        for row in item["examples"]:
            ref = baseline_rows[row["idx"]]
            prompt_rows.append({
                "idx": row["idx"], "gold_mean_nll": row["gold_answer_mean_nll"],
                "nll_delta_vs_dense": row["gold_answer_mean_nll"] - ref["gold_answer_mean_nll"],
                "gold_top1": f"{row['gold_token_top1_count']}/{row['gold_token_top1_total']}",
                "first_target_top1": row["first_target_top1"],
                    "mean_attention_output_cosine_vs_dense": sum(
                        F.cosine_similarity(
                            torch.tensor(row["attention_output_at_answer_queries_by_layer"][str(i)]),
                            torch.tensor(ref["attention_output_at_answer_queries_by_layer"][str(i)]),
                            dim=-1).item()
                        for i in range(N_LAYERS)
                    ) / N_LAYERS,
            })
        summary.append({"name": item["name"], "mode": item["mode"],
                        "dense_layers": item["dense_layers"],
                        "mean_gold_nll": sum(x["gold_answer_mean_nll"] for x in item["examples"]) / len(item["examples"]),
                        "mean_gold_token_top1_fraction": sum(x["gold_token_top1_count"] for x in item["examples"]) / sum(x["gold_token_top1_total"] for x in item["examples"]),
                        "first_target_top1_count": sum(x["first_target_top1"] for x in item["examples"]),
                        "mean_attention_output_cosine_vs_dense": sum(x["mean_attention_output_cosine_vs_dense"] for x in prompt_rows) / len(prompt_rows),
                        "per_prompt": prompt_rows})
    # The wrapper path should reproduce the exp0 reference closely; record
    # differences as a validation rather than silently treating it as a new arm.
    dense_wrapper = next(x for x in summary if x["name"] == "dense_wrapper")
    dense_ref = next(x for x in summary if x["name"] == "dense_flash")
    wrapper_nll_delta = dense_wrapper["mean_gold_nll"] - dense_ref["mean_gold_nll"]
    h.write_json(OUT / "teacher_forced_summary.json", {
        "results": summary, "dense_wrapper_mean_nll_delta_vs_exp0": wrapper_nll_delta,
        "dense_wrapper_matches_exp0": abs(wrapper_nll_delta) < 0.25,
        "selection": "No selector ranking; mechanistic diagnostic only.",
        "limitations": [
            "Teacher-forced next-token likelihood is not free-running exact-retrieval accuracy.",
            "Answer-aware oracle routes are diagnostic only.",
            "One seed/depth and three prompts do not support generalization or scaling claims.",
            "The layer hybrids locate sensitivity but are not proposed as production methods.",
        ],
    })
    print("ROUND3_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
