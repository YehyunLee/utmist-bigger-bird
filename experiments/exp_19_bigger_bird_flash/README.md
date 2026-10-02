# Bigger Bird v2: proposal-inspired routing and Triton sparse FlashAttention

Working directory: `/home/thomas7/projects/def-guerzhoy/thomas7/utmist-bigger-bird`.

This is experiment 19. It preserves exp 5/15 as historical experiments and
implements an inference variant of the Bigger Bird proposal on frozen
DeepSeek-R1-Distill-Llama-8B weights.

## Selection and scaling

Three components form one deduplicated token set:

1. **Content globals:** pick the most salient token in each temporal coverage
   stratum. Salience combines low-dimensional key norm and novelty relative to
   the adjacent key. The positions depend on content; they are not fixed front
   anchors. This is a parameter-free coverage heuristic, not a trained gate or
   a proven facility-location optimizer.
2. **Selective wide local attention:** rank candidates within a neighborhood
   that grows with context and retain a one-step MMR shortlist. MMR penalizes
   similarity to the best relevant candidate while always preserving that
   candidate. An exact 128-token recent backbone maintains immediate LM context.
3. **Content-biased long links:** relevance plus one-step MMR over a 2*k shortlist
   replaces uniform random teleports. No random links are used in this variant.

For context length `n`, default budgets are:

| Context | Wide candidate window | Local selections | Globals | Long links | Exact recent |
|---|---:|---:|---:|---:|---:|
| 4K | 1,024 | 128 | 32 | 512 | 128 |
| 16K | 1,024 | 128 | 32 | 512 | 128 |
| 32K | 2,048 | 128 | 32 | 512 | 128 |
| 64K | 4,096 | 256 | 32 | 1,024 | 128 |
| 128K | 8,192 | 512 | 64 | 2,048 | 128 |

Counts are caps; overlap, padding, and visible-prefix constraints reduce actual
counts. The wide window is a candidate neighborhood, not that many exact
attention keys. The schedule is length-adaptive; uncertainty-driven budgets
are not implemented. The model's position limit remains 128K.

## Execution

`kernels/bigger_bird_flash.py` implements tiled tensor-core attention and
FlashAttention-style online softmax in Triton. It reads selected keys directly,
without materializing `[heads, queries, budget, head_dim]` gathered K/V tensors.
Local/global/long links share one softmax; overlaps with the exact recent
window are removed. Causality uses original positions, including cached decode.
The default query tile is 64, selected from measured 32/64 comparisons.
Q is already scaled by the Llama projections, so the kernel uses `scale=1.0`.

`kernels/bigger_bird_routing.py` fuses relevance/salience scoring and shortlist
MMR reranking in Triton. Top-k and sorting still use PyTorch GPU operations.
This is a custom sparse FlashAttention-style kernel, not a call to the stock
Dao-AILab FlashAttention API on a packed sequence.

The KV cache retains **all unexpanded GQA keys/values**. Sparse selection limits
attention work, not persistent cache size. Memory therefore still grows with
context. Capped attention budgets alone do not prove sub-quadratic total work;
routing cost must also be counted. In particular, chunk routing scans the
prefix repeatedly and has quadratic worst-case routing cost for fixed chunks.

## Routing modes and cache semantics

- `last_query`: the fastest full-prompt research path, matching the existing
  repository's use of a final query to route earlier prompt queries. Attention
  edges are causal-masked, but the *routing decisions* depend on later prompt
  content. This mode is not prefix-invariant. Cached generation freezes prompt
  representations after prefill and reroutes only the current decode query;
  it is a different procedure from rerouting the whole prompt at every step.
  Report it as full-prompt routing, not standard streaming causal attention.
- `causal_chunk`: select using the first query and visible prefix of each chunk.
  Budgets depend on that prefix. Cached decode reuses the chunk's selected set
  and refreshes it at the next chunk boundary, matching full recomputation.
  This mode passes prefix-invariance and cache-equivalence tests. It can lose
  accuracy because long links are shared between queries in a chunk.

Both modes include the proposal's content globals, selective wider windows,
and diversity-aware long links. Neither is the original fixed-pattern Google
BigBird implementation. The current default is the explicitly labelled
full-prompt `last_query` research path; the causal mode is an ablation.

## Tokenizer and benchmark validity

In the installed Transformers 5.15.1 environment, `AutoTokenizer` resolves the
checkpoint to a Llama/SentencePiece tokenizer and changes its saved byte-level
BPE processing. A round trip removes spaces. Experiment 19 loads the original
`tokenizer.json` backend with `PreTrainedTokenizerFast` and verifies an exact
round trip. Other experiments' loading behavior was left intact for historical
reproducibility. They need a tokenizer audit before comparing fresh quality runs
from this environment with earlier results.

The first smoke attempt using the incorrect tokenizer is preserved under
`benchmarks/bigger_bird_flash/invalid_tokenizer/` and is invalid for quality.
Results in `smoke_correct_tokenizer.json`, `smoke_cached.json`, and the final
fused run use the checkpoint's saved BPE tokenizer. The first corrected run
stopped during a 16K cached dense OOM; completed examples are retained.

The smoke test is custom RULER-style exact-number retrieval, **not the official
RULER benchmark suite**. It builds actual tokenizer-length prompts, pairs model
variants on the same prompts, and scores the full seven-digit answer. It
generates exactly six tokens per run. All generation lengths are warmed so
compilation is excluded. Dense uses native Hugging Face SDPA with the Flash
backend forced; cached comparisons use the same full GQA cache policy.

## Run on Narval

The tested environment has PyTorch 2.13.0, Transformers 5.15.1, and the Alliance
Triton 3.6.0 wheel. GPU checks must run inside a Slurm allocation.

```bash
cd /home/thomas7/projects/def-guerzhoy/thomas7/utmist-bigger-bird
sbatch scripts/run_bigger_bird_smoke.sbatch
```

The batch runs numerical/selection checks, cache equivalence, attention timing
(including routing), and paired full-model retrieval. It selects a query tile
size using measured complete sparse-attention time. It also tests a context
length off the exact routing-chunk boundary. Outputs are stored in
`benchmarks/bigger_bird_flash/`; Slurm output is in `/scratch/thomas7/`.

Experiment 19 is also registered in `eval/ruler_llama/run_generative.py`. That
legacy runner still uses uncached generation and its existing reduced-label
RULER-style dataset; use the new paired smoke runner for exact-number/cache
comparisons. Historical experiment results have not been overwritten.

## Verification

- `tests/test_bigger_bird_flash.py`: FP16/BF16 agreement with an independent
  unique-union reference; masks, padding, overlap, cached query offsets,
  selector uniqueness, prefix invariance, and fused MMR against its PyTorch
  formula.
- `tests/test_bigger_bird_cache.py`: cached vs full recomputation on a tiny real
  Llama, before/at/after a routing-chunk boundary.
- `scripts/benchmark_bigger_bird_flash.py`: native forced dense FlashAttention
  vs kernel-only and routing-plus-attention time, with GQA and BF16.
- `scripts/smoke_bigger_bird_flash.py`: paired frozen 8B-model retrieval and
  complete generation timings. Small sample counts are smoke evidence, not
  established long-context accuracy.

## Completed small runs (2026-09-26)

On A100 MIG 3g.20gb, cached full-prompt experiment 19 took 1.796 s vs
1.694 s dense at 8K, 3.348 s vs 3.653 s at 16K, and 3.406 s vs 3.730 s
at 16,521 tokens. Both variants retrieved all six full seven-digit answers.
The observed 16K speedup is 1.091x; sample counts are small.

Routing-plus-attention synthetic speedups at 8K/16K/32K/64K/128K were
2.02x/4.21x/8.75x/10.99x/11.87x. These are attention timings, not full-model
speed or long-context accuracy. Full-model accuracy beyond 16K is untested.

The prefix-invariant 4,096-token causal-chunk ablation retrieved 4/6 answers:
both off-boundary 16,521-token examples failed. It remains experimental.
Detailed timings, caveats, and reproduction are in
`benchmarks/bigger_bird_flash/summary.md`; raw results are retained alongside it.

Final GPU checks (job 4056041) passed with both 32/64-query tiles, additive
mask/short-context checks, and BF16 cache-equivalence max logit error 0.003906.
