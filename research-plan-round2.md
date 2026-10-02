# Bigger Bird R&D round 2

The first six-example screen at 127,218–127,733 actual model tokens found 0/6 exact retrieval for every sparse arm, against 5/6 for matched dense. Distinct block selection fixed duplicate slots but did not restore retrieval. This round tests routing-query choice and separates missed answer routes from other failures.

Remote checkout: `/scratch/thomas7/bb-rd-20260928`, base commit `eb7650a9046946514a4e4185c6a26539fb40e804`. Round-1 sources and artifacts remain intact. New source files are hashed with inherited attention, routing, evaluation and harness sources. No original checkout, dashboard or branch is changed.

## Predeclared screen

Use the same first six prompts: source bytes 512,000, depth 0.5, seed 42, first six of 128 examples, frozen DeepSeek-R1-Distill-Llama-8B checkpoint, BF16 greedy generation, up to ten output tokens, no KV cache, H100 MIG 3g.40GB. Each arm warms all its prompts before timing those same prompts. Record exact full-number and last-digit scores, actual tokens and hashes, first-forward prefill, total task runtime, output/forward counts, memory and untimed per-layer routing traces.

| Arm | Purpose | Prompts | Benchmark eligible? |
|---|---|---:|---|
| dense_flash | Fresh paired reference | 6 | Reference |
| coverage_control | Matched round-1 distinct blocks, 12.5% salience coverage | 6 | Control |
| oracle_all_layers | Force a 128-token context centered on the gold statement into every head of all 32 layers | 3 | No; answer-aware diagnostic |
| oracle_last8_layers | Same forced context only in layers 24–31 | 3 | No; answer-aware diagnostic |
| rotated_question_max | Maximum cosine across up to eight visible question-token queries plus latest query | 6 | Yes |
| raw_question_max | Same score using Q/K before RoPE; actual attention uses normal rotated Q/K | 6 | Yes |
| raw_question_max_local512 | Raw question routing with exact recent backbone widened from 128 to 512 | 6 | Yes; backbone change explicitly recorded |

All sparse arms use the original square-root routed budget (about 4,096 slots at this context), 32-token blocks, distinct coverage/local/long-range selections and the same Triton union-softmax attention. Ordinary selectors receive only the question boundaries derived from visible prompt text, never answer offsets. The forced diagnostics replace ordinary slots, preserve total route budget and remain sparse. They cannot be selected for confirmation or counted as benchmark results.

Question-aware routing is task-specific. It is globally conditioned on the question near the prompt end; attention edges are causal but earlier routing can depend on later queries. No prefix-invariance, arbitrary-task, or KV-cache-equivalence claim is made.

## Decision rule

An ordinary selector qualifies only with at least 4/6 exact numbers, higher exact accuracy than the fresh coverage control, and shorter mean first-forward prefill than fresh dense. Rank qualifying arms by exact count, then prefill time. Oracles are excluded. Compare oracle scores against dense on the same three indices; a failed oracle does not alone identify the failure mechanism, and successful forced retention is not a valid deployable improvement.

If one qualifies, the conditional job measures fresh dense and the selected arm at 261,000 and 512,000 source bytes (roughly 65K and 127K actual tokens), all 30 prompts warmed then the same 30 timed. Report examples 6–29 separately because examples 0–5 tuned the selector at the longer context. These held-out examples share the generator, seed and depth, so they are a limited within-task confirmation, not a broad generalization test. Otherwise confirmation writes a skipped artifact and exits.

GPU reference checks must pass before experiment work: independent multiquery cosine block maxima and anchors, full unique route budgets, constant-budget forced context, signal retention under maximum pooling, pre-RoPE projection capture versus properly rotated causal union attention, cleanup of captured tensors/hooks, and rejection of gold context by ordinary candidates. Preserve failures and partial outputs. Do not submit unattended replacement jobs.

## Reporting

Copy raw final/progress JSON, manifests, source fingerprints and logs locally. Independently verify counts/scores, prompt/source hashes, warm/timed predictions, context lengths, output/forward counts, timing and memory. Label diagnostic and tuning evidence separately from confirmation. Report accuracy alongside prefill and task runtime; two context points cannot establish subquadratic end-to-end scaling.
