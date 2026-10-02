"""Matched synthetic RULER-style NIAH prompts (mirror of the Round 4 builder).

Prompt text, answers and BOS-prefixed token IDs reproduce the CCDB Round 4
screens exactly; `prepare(..., verify=True)` checks token hashes against the
saved manifests when one exists for the requested source-byte length.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from tokenizers import processors
from transformers import PreTrainedTokenizerFast

from eval.lra_llama.lra_llama_dataset import _ids_to_text
from eval.ruler.ruler_dataset import build_ruler_dataset

MODEL_PATH = os.path.join(os.environ.get("SCRATCH", "/vol"), "models", "DeepSeek-R1-Distill-Llama-8B")
MANIFESTS = Path(__file__).parent / "manifests"
MANIFEST_KEYS = ("idx", "label", "answer", "tokens", "token_sha256", "source_sha256", "spans", "question_tokens")


def load_tokenizer(model_path=MODEL_PATH):
    """tokenizer.json backend + explicit BOS (fdf13e8); verifies exact round trip."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    bos = tokenizer.bos_token
    tokenizer.backend_tokenizer.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(single=f"{bos}:0 $A:0", pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
                                      special_tokens=[(bos, tokenizer.bos_token_id)]),
    ])
    probe = "One of the special magic numbers is 2940341.\nWhat is it?"
    if tokenizer.decode(tokenizer(probe, add_special_tokens=False)["input_ids"]) != probe:
        raise RuntimeError("Tokenizer round-trip failed")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 131072
    return tokenizer


def prepare(tokenizer, seq_bytes, count, depth=0.5, seed=42, start=0, verify=True):
    data = build_ruler_dataset(task="niah", seq_len=seq_bytes, needle_depth=depth,
                               train_samples=10, eval_samples=max(128, start + count), seed=seed)["validation"]
    rows = []
    for idx in range(start, start + count):
        text = _ids_to_text(data[idx]["input_ids"]).rstrip() + " The answer is:"
        question = re.search(r"What is the special magic ([\w-]+)\?", text)
        needle = re.search(r"The special magic " + re.escape(question[1]) + r" is: (\d+)\.", text)
        assert needle and int(needle[1][-1]) == int(data[idx]["labels"]), (seq_bytes, idx)
        enc = tokenizer(text, return_offsets_mapping=True, truncation=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        assert len(ids) + 10 <= 131072, (seq_bytes, idx, len(ids))
        hits = lambda a0, b0: [i for i, (a, b) in enumerate(offsets) if b > a0 and a < b0 and b > a]
        spans = {k: [min(h), max(h) + 1] for k, h in (("needle", hits(needle.start(), needle.end())),
                                                       ("number", hits(needle.start(1), needle.end(1))))}
        q = hits(question.start(), question.end())
        rows.append({"idx": idx, "text": text, "label": int(data[idx]["labels"]), "answer": needle[1],
                     "tokens": len(ids), "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "spans": spans, "question_tokens": [min(q), max(q) + 1], "depth": depth, "seed": seed})
    manifest = MANIFESTS / f"bytes{seq_bytes}.json"
    if verify and depth == 0.5 and seed == 42 and manifest.exists():
        saved = {r["idx"]: r for r in json.loads(manifest.read_text())}
        for r in rows:
            if r["idx"] in saved:
                assert {k: r[k] for k in MANIFEST_KEYS} == saved[r["idx"]], f"prompt {r['idx']} differs from CCDB manifest"
    return rows
