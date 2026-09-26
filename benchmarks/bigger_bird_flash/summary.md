# Bigger Bird + growing context + Triton FlashAttention: implemented results

Implemented on Narval in `/home/thomas7/projects/def-guerzhoy/thomas7/utmist-bigger-bird`, as experiment 19. Frozen DeepSeek-R1-Distill-Llama-8B; A100 MIG 3g.20gb; BF16; batch size 1. Existing experiments and prior user edits were preserved.

## Observed full-model latency and retrieval

Each timing includes prefill and exactly six generated tokens with KV caching. Model loading and kernel compilation are excluded. Native dense attention is forced to use PyTorch's FlashAttention backend. Both variants receive identical prompts with actual tokenizer lengths. Two examples per length, at 25% and 75% depth, require the full seven-digit answer. This is a custom retrieval smoke test, not the official RULER suite or a broad accuracy evaluation.

| Input tokens | Dense FA | Bigger Bird full-prompt | Dense / Bigger Bird | Dense correct | Bigger Bird correct |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1.694 s | 1.796 s | 0.943× | 2/2 | 2/2 |
| 16,384 | 3.653 s | 3.348 s | 1.091× | 2/2 | 2/2 |
| 16,521 | 3.730 s | 3.406 s | 1.095× | 2/2 | 2/2 |

At 16,384 tokens, observed generation time fell by 8.3% (1.091× speedup). At 8K, routing and integration overhead still outweigh the attention savings: Bigger Bird is about 6% slower. These are small, warmed smoke runs, not confidence intervals or production latency guarantees. The full-prompt version retrieved all six numbers correctly.

The faster full-prompt router uses the final prompt query to select keys shared by earlier prompt queries. Attention edges remain causal, but routing decisions depend on later prompt content. It therefore does **not** have standard streaming causal semantics. Cached generation freezes prompt states after prefill and reroutes the current decode query.

The separately tested prefix-invariant `causal_chunk` ablation (4,096-token routing chunks) passed numerical cache equivalence but retrieved only 4/6 numbers. It passed at 8K and 16K, then failed both 16,521-token examples. Correctness of the kernel/cache does not establish task accuracy. This mode remains experimental; the full-prompt mode is the default research path.

## Attention benchmark, including routing

Synthetic Q/K/V matching 32 query heads, 8 KV heads, and head dimension 128. Dense FA and sparse attention use original causal positions. Complete sparse time includes selection plus attention, measured together; it need not equal the sum of separately measured stages. Query tile 64 was chosen against tile 32 using the 8K and 16K measurements. Three timed repeats after warmup.

| Context | Dense FA | Routing + sparse attention | Speedup |
|---:|---:|---:|---:|
| 4,096 | 1.94 ms | 2.04 ms | 0.95× |
| 8,192 | 7.08 ms | 3.50 ms | 2.02× |
| 16,384 | 27.35 ms | 6.49 ms | 4.21× |
| 32,768 | 106.87 ms | 12.21 ms | 8.75× |
| 65,536 | 422.91 ms | 38.48 ms | 10.99× |
| 131,072 | 1685.32 ms | 142.04 ms | 11.87× |

64K/128K results measure attention on synthetic tensors, **not** complete 8B-model inference or accuracy. The allocated 20GB GPU partition limited the full-model runs to approximately 16K. Persistent KV-cache size is unchanged; all GQA keys and values remain stored. Sparse attention alone does not solve cache memory growth.

## How the implementation works

1. **Content-selected globals:** salient/novel keys chosen within temporal coverage strata. Positions depend on content, with no fixed front anchors.
2. **Selective wide local window:** consider a wider neighborhood as context grows, retain relevant candidates, and penalize redundancy with a one-step MMR filter. Keep 128 recent tokens exact for continuity.
3. **Content-biased long links:** relevance plus the same diversity filter replaces uniform random token selection.
4. **One sparse FlashAttention operation:** deduplicate the selected union and compute a joint softmax with tiled tensor-core operations and online normalization in Triton. Keys are read directly; no per-query gathered K/V tensor is materialized. The relevance/salience and diversity reranking stages also use Triton. Top-k and sorting remain PyTorch GPU operations.

This is a proposal-inspired inference implementation with parameter-free gates and frozen weights. It has no learned routing gate or full facility-location optimizer. The attention implementation is a custom Triton FlashAttention-style kernel, rather than the stock Dao-AILab API.

| Context | Wide candidate window | Selected local | Selected globals | Long-link budget | Exact recent |
|---:|---:|---:|---:|---:|---:|
| 4K–16K | 1,024 | 128 | 32 | 512 | 128 |
| 32K | 2,048 | 128 | 32 | 512 | 128 |
| 64K | 4,096 | 256 | 32 | 1,024 | 128 |
| 128K | 8,192 | 512 | 64 | 2,048 | 128 |

Budgets are caps, with duplicates/invalid candidates removed. Growth is based on context length, not query uncertainty. Caps stop increasing at 128K. Fixed-chunk routing repeatedly scans prefixes and has quadratic worst-case routing work; bounded attention budgets do not imply a total complexity guarantee.

## Validity fixes and verification

The installed Transformers 5.15.1 `AutoTokenizer` altered the checkpoint's byte-level processing and failed an exact text round trip. Experiment 19 and the paired runner now load the saved `tokenizer.json` backend directly and verify the round trip. The first incorrect-tokenizer run is retained separately and excluded from quality claims. The earlier corrected smoke stopped at a cached dense 16K memory error; enabling expandable allocation allowed the final matched comparison to complete.

GPU checks compare FP16/BF16 sparse outputs against an independent dense unique-union reference, including padding, overlaps, original cached positions, selection deduplication, prefix invariance, and fused diversity ranking. A tiny real Llama checks cached/full-recomputation logits across chunk boundaries. The final 64-query-tile verification also covers short contexts and boolean versus additive masks.

Benchmark job 4055569 completed successfully with exit 0. Final check job 4056041 also completed successfully with exit 0. Both 32- and 64-query tiles passed the independent-reference checks. The boolean/additive mask and short-context checks passed. Cached/full-recomputation logits differed by at most 0.003906 in BF16, within the checked tolerance. The full log is in [final_checks.log](final_checks.log).

## Files and reproduction

Main code: `kernels/bigger_bird_flash.py`, `kernels/bigger_bird_routing.py`, and `experiments/exp_19_bigger_bird_flash/`. Experiment 19 is registered in the existing generative runner. That legacy runner retains uncached generation and reduced-label scoring; use the new paired runner for the comparisons reported here.

```bash
cd /home/thomas7/projects/def-guerzhoy/thomas7/utmist-bigger-bird
sbatch scripts/run_bigger_bird_smoke.sbatch
```

A five-minute numerical/cache check is available separately:

```bash
sbatch scripts/run_bigger_bird_checks.sbatch
```

Remote raw results: `benchmarks/bigger_bird_flash/`. Local copies next to this report: [model smoke](smoke_fused_cached.json), [attention timings](microbench_fused64.json), [tile-32 comparison](microbench_fused32.json), and [earlier cached smoke](smoke_cached.json).

## Next work suggested by the evidence

A larger GPU allocation would enable paired 32K/64K/128K full-model quality tests with multiple seeds, depths, and harder retrieval tasks. Test budget growth against fixed budgets to establish whether scaling actually improves accuracy. The strict causal router needs query-sensitive refresh or a better shared query representation before it can be promoted. Reducing GQA expansion, repeated selection launches, and decode routing overhead is the next latency opportunity. Preserve paired native dense FA as the baseline and include all routing costs.
