# Bigger Bird long-context R&D — 28 September 2026

Goal: recover exact-number retrieval at about 128K actual checkpoint tokens
while maintaining useful speed relative to matched dense FlashAttention.
All changes live in `/scratch/thomas7/bb-rd-20260928`; the previous scratch
checkout, main home checkout and dashboard remain preserved.

## Experiment arms

| Arm | Change relative to previous sqrt_spans32 |
|---|---|
| dense_flash | Paired dense FlashAttention control |
| sqrt_spans32 | Exact preserved token-anchor expansion wrapper |
| block_distinct | Max-token relevance per 32-token block, disjoint branch selection, refill invalid slots; original global allocation |
| block_coverage | Distinct selection plus 12.5% of blocks reserved for salience coverage across temporal strata |
| block_coverage_1p5x | Same coverage selector with 1.5× square-root budget |
| block_prefix4 | Same coverage selector with at most four visible-prefix routing groups |

All sparse arms retain the 128-token exact recent backbone, content-aware
coverage, wide-local MMR, long-range MMR, frozen weights, and the existing
single-softmax Triton sparse FlashAttention kernel. Global block relevance
uses the strongest valid token, not a block mean. The new metadata reduction
and block expansion are Triton kernels. Top-k and MMR act on block summaries.
The former MMR is a one-step penalty against the strongest candidate, not
the original proposal's full learned gate or a facility-location guarantee.

## Evaluation

GPU tests must pass before screening. Tests compare block scores to an
independent reference, enforce distinct/full routing budgets, check earlier
prefix routing against modified future inputs, and compare sparse Flash
outputs to an explicit masked softmax over the identical key union.

Screen: first six of the same 128 NIAH candidates at source bytes 512000,
depth 0.5, seed 42. Warm all six, then time those six. This is a selector
tuning experiment, not the final benchmark or the historical grid.

Candidate must achieve at least 4/6 exact numbers, beat the old span control,
and have shorter first-forward prefill time than matched dense. Rank by
exact accuracy, then prefill time. If none qualifies, do not run expensive
confirmation; report the negative result and routing evidence.

Conditional confirmation: selected candidate and a fresh dense control at
source bytes 261000 (~65K tokens) and 512000 (~127K tokens). Warm all 30
prompts, then time the same 30. Report both all-30 performance and the
untuned examples 6–29 (24 examples) separately at 128K.

Protocol: repository synthetic NIAH, not the full official RULER suite;
DeepSeek-R1-Distill-Llama-8B, checkpoint BPE tokenizer, BF16, greedy,
max_new_tokens=10, use_cache=False, H100 MIG 3g.40GB. No truncation allowed.
Record source/token hashes, actual token lengths, generated token IDs/counts,
forward counts/context lengths, exact-number and last-digit scores,
per-example wall time, first-forward CUDA-event time, memory, source hashes,
job IDs and attention settings. Runtime excludes diagnostic calculations.

Routing diagnostics run only during the untimed first forward of each warm
prompt. Gold offsets are never supplied to a selector. They measure unique
active tokens, number-span retention and statement coverage per layer/head.

## Limits

Last-query routing retains the old future-dependent route policy, even
though attention edges are causal. Prefix routing is future-independent
within a fixed group layout; its layout changes with total context length,
so this is not a claim of cached-generation equivalence. No KV cache is used.
Fixed routing groups and square-root budgets target O(N^1.5) attention work;
end-to-end runtime and routing overhead must be measured, not assumed.
One seed and six tuning examples are narrow evidence. Winning at this NIAH
task does not establish general long-context capability.
