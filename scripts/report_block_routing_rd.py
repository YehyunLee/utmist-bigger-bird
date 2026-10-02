"""Verify saved R&D artifacts and write a concise comparison report."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    results = []
    checks = []
    for folder in sorted(root.glob("*_*/")):
        for path in sorted(folder.glob("*_bytes*.json")):
            if "progress" in path.name or "manifest" in path.name:
                continue
            x = json.loads(path.read_text())
            if "examples" not in x:
                continue
            assert len(x["examples"]) == x["n_examples"] == x["warmup_examples"]
            assert len(x["diagnostic_examples"]) == x["n_examples"]
            assert sum(e["exact_correct"] for e in x["examples"]) == x["exact_number_correct"]
            assert sum(e["last_digit_correct"] for e in x["examples"]) == x["last_digit_correct"]
            assert abs(sum(e["time_seconds"] for e in x["examples"]) - x["time_seconds"]) < 1e-6
            manifest = json.loads((folder / f"prompt_manifest_bytes{x['seq_len_bytes']}.json").read_text())
            assert [m["token_sha256"] for m in manifest] == [e["token_sha256"] for e in x["examples"]]
            assert all(w["generated_ids"] == e["generated_ids"] for w, e in zip(x["diagnostic_examples"], x["examples"]))
            assert all(len(e["generated_ids"]) == e["generated_token_count"] == e["forward_count"] for e in x["examples"])
            assert x["max_new_tokens"] == 10 and x["use_cache"] is False and x["dtype"] == "bfloat16"
            assert x["peak_memory_gb"] > 0
            assert min(m["tokens"] for m in manifest) == x["actual_tokens_min"]
            assert max(m["tokens"] for m in manifest) == x["actual_tokens_max"]
            results.append((path, x))
            checks.append({"file": str(path), "verified": True})
    assert results, "No completed results yet"
    groups = {}
    for path, x in results:
        groups.setdefault((path.parent.name, x["seq_len_bytes"]), []).append(x)
    lines = ["# Bigger Bird long-context R&D results", "",
        "Source-byte sizes and actual model-token ranges are shown separately. Accuracy means exact full-number retrieval. All methods warm every measured prompt before timing it. Gold answer offsets are used only for untimed routing diagnostics.", ""]
    for (stage, seq), runs in groups.items():
        first = runs[0]
        assert all([e["token_sha256"] for e in x["examples"]] == [e["token_sha256"] for e in first["examples"]] for x in runs)
        assert all(x["sources"] == first["sources"] for x in runs)
        dense = next((x for x in runs if x["name"] == "dense_flash"), None)
        lines += [f"## {stage}: {first['actual_tokens_min']:,}–{first['actual_tokens_max']:,} actual tokens ({seq:,} source bytes)", "",
            "| Selector | Exact score | Last digit | Wall s/prompt | Prefill s/prompt | Prefill speed vs dense | Mean generated tokens | Peak GB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for x in sorted(runs, key=lambda z: (z["name"] != "dense_flash", -z["exact_number_correct"], z["prefill_seconds"])):
            n = x["n_examples"]
            speed = f"{dense['prefill_seconds'] / x['prefill_seconds']:.2f}×" if dense else "pending"
            lines.append(f"| {x['name']} | {x['exact_number_correct']}/{n} | {x['last_digit_correct']}/{n} | {x['time_seconds']/n:.2f} | {x['prefill_seconds']/n:.2f} | {speed} | {statistics.mean(e['generated_token_count'] for e in x['examples']):.2f} | {x['peak_memory_gb']:.2f} |")
        if first["stage"] == "confirm":
            lines += ["", "Examples 0–5 were used for selector tuning at 128K. Untuned examples 6–29:"]
            for x in runs:
                lines.append(f"- {x['name']}: {x['held_out_exact_correct']}/{x['held_out_examples']} exact.")
        sparse = [x for x in runs if x["exp"] == 19]
        if sparse:
            lines += ["", "Untimed routing observations (last routing group, first forward):", "",
                "| Selector | Mean unique tokens / slots | Layers retaining full number in any head | Full-number retention, mean head fraction |", "|---|---:|---:|---:|"]
            for x in sparse:
                layer_rows = [a for e in x["diagnostic_examples"] for a in e["routing_by_layer"]]
                lines.append(f"| {x['name']} | {statistics.mean(a['unique_tokens_mean'] for a in layer_rows):.1f} / {statistics.mean(a['allocated_slots'] for a in layer_rows):.1f} | {statistics.mean(a['number_full_in_any_head'] for a in layer_rows):.1%} | {statistics.mean(a['number_full_head_fraction'] for a in layer_rows):.1%} |")
        lines.append("")
    confirmation = [x for _, x in results if x["stage"] == "confirm"]
    by_name = {}
    for x in confirmation:
        by_name.setdefault(x["name"], {})[x["seq_len_bytes"]] = x
    if any(len(xs) == 2 for xs in by_name.values()):
        lines += ["## Measured 64K → 128K growth", ""]
        for name, xs in by_name.items():
            if 261000 in xs and 512000 in xs:
                a, b = xs[261000], xs[512000]
                lines.append(f"- {name}: prefill {b['prefill_seconds']/a['prefill_seconds']:.2f}×; total generation wall time {b['time_seconds']/a['time_seconds']:.2f}×; exact {a['exact_number_correct']}/30 → {b['exact_number_correct']}/30.")
    lines += ["", "## Interpretation limits", "",
        "Six screening prompts are tuning evidence. Confirmation reports the remaining 24 separately. The task is repository synthetic NIAH, not the full official RULER suite. Generated lengths vary, so use first-forward prefill for fixed-input speed and read generation wall time alongside output counts. Two context lengths and one seed do not prove asymptotic scaling or general task quality. Earlier completed runs are preserved and are not used as a substitute for the fresh paired dense control.", ""]
    (root / "results-report.md").write_text("\n".join(lines))
    (root / "verification.json").write_text(json.dumps({"checks": checks, "groups": len(groups), "prompt_hash_match": True}, indent=2))
    print(root / "results-report.md")


if __name__ == "__main__":
    main()
