# Bigger Bird v3 Modal R&D results

## Confirmation (30 prompts per context; prompts 0-5 were used for tuning at ~127K)

| context | depth | arm | exact all-30 | exact held-out 6-29 | last digit | mean prefill s |
|---|---:|---|---:|---:|---:|---:|
| ~65K | 0.5 | Dense FlashAttention | 30/30 | 24/24 | 30/30 | 5.02 |
| ~128K | 0.5 | Dense FlashAttention | 28/30 | 22/24 | 28/30 | 16.25 |
| ~65K | 0.5 | sink64+win4096+vert512+tail256 | 30/30 | 24/24 | 30/30 | 3.10 |
| ~128K | 0.5 | sink64+win4096+vert512+tail256 | 28/30 | 22/24 | 28/30 | 6.10 |
| ~65K | 0.5 | sink64+win16384+vert512+tail256 | 30/30 | 24/24 | 30/30 | 4.41 |
| ~128K | 0.5 | sink64+win16384+vert512+tail256 | 30/30 | 24/24 | 30/30 | 8.61 |
| ~128K | 0.1 | Dense FlashAttention | 28/30 | 23/24 | 28/30 | 16.39 |
| ~128K | 0.1 | sink64+win16384+vert512+tail256 | 30/30 | 24/24 | 30/30 | 8.56 |

## Harder RULER NIAH variants (12-24 prompts per cell; none used for tuning)

| task | description | context | dense exact | sparse exact | dense recall | sparse recall |
|---|---|---:|---:|---:|---:|---:|
| niah_single_2 | single needle, essay haystack | ~32K | 12/12 | 12/12 | 1.00 | 1.00 |
| niah_single_2 | single needle, essay haystack | ~64K | 12/12 | 12/12 | 1.00 | 1.00 |
| niah_single_2 | single needle, essay haystack | ~128K | 17/24 | 11/24 | 0.71 | 0.46 |
| niah_single_3 | single needle, UUID value | ~128K | 0/12 | 0/12 | 0.00 | 0.00 |
| niah_multikey_1 | 4 keys, retrieve 1 | ~32K | 12/12 | 12/12 | 1.00 | 1.00 |
| niah_multikey_1 | 4 keys, retrieve 1 | ~64K | 9/12 | 11/12 | 0.75 | 0.92 |
| niah_multikey_1 | 4 keys, retrieve 1 | ~128K | 4/12 | 1/12 | 0.33 | 0.08 |
| niah_multikey_3 | UUID keys in a haystack of UUID distractors | ~128K | 0/12 | 0/12 | 0.00 | 0.00 |
| niah_multivalue | 4 values for one key | ~32K | 4/12 | 5/12 | 0.73 | 0.77 |
| niah_multivalue | 4 values for one key | ~64K | 3/12 | 6/12 | 0.48 | 0.67 |
| niah_multivalue | 4 values for one key | ~128K | 0/12 | 0/12 | 0.17 | 0.17 |
| niah_multiquery | 4 keys queried at once | ~32K | 12/12 | 12/12 | 1.00 | 1.00 |
| niah_multiquery | 4 keys queried at once | ~64K | 8/12 | 8/12 | 0.92 | 0.92 |
| niah_multiquery | 4 keys queried at once | ~128K | 0/12 | 0/12 | 0.23 | 0.19 |

## Context scaling (6 prompts / length, depth 0.5, all prompts warmed, KV cache)

| tokens | dense prefill s / exact | Bigger Bird v3, window 4K prefill s / exact | Bigger Bird v3, window 16K (final) prefill s / exact |
|---:|---:|---:|---:|
| 7965 | 0.27 / 6/6 | 0.38 (0.72x) / 6/6 | 0.27 (1.01x) / 6/6 |
| 15963 | 0.64 / 5/6 | 0.74 (0.86x) / 6/6 | 0.63 (1.01x) / 5/6 |
| 32449 | 1.70 / 6/6 | 1.52 (1.11x) / 6/6 | 2.13 (0.80x) / 6/6 |
| 65088 | 5.03 / 6/6 | 3.11 (1.61x) / 6/6 | 4.35 (1.15x) / 6/6 |
| 127545 | 16.22 / 6/6 | 6.12 (2.65x) / 6/6 | 8.50 (1.91x) / 6/6 |

## Empirical scaling exponent (prefill ~ tokens^k, fitted on 32K-127K)

| arm | k |
|---|---:|
| Dense FlashAttention | 1.65 |
| Bigger Bird v3, window 4K | 1.02 |
| Bigger Bird v3, window 16K (final) | 1.01 |

## Attention-only cost per layer (synthetic Llama-8B shapes, BF16, H100; not a quality result)

| tokens | dense | sparse | speedup |
|---:|---:|---:|---:|
| 32768 | 13 ms | 30 ms | 0.43x |
| 65536 | 56 ms | 61 ms | 0.92x |
| 131072 | 234 ms | 125 ms | 1.88x |
| 262144 | 1049 ms | 251 ms | 4.18x |
| 524288 | 4422 ms | 508 ms | 8.70x |
| 1048576 | 18422 ms | 1038 ms | 17.76x |

## Needle depth @~127K (6 prompts / depth)

| depth | Dense FlashAttention | Bigger Bird v3, window 4K | Bigger Bird v3, window 16K (final) |
|---:|---:|---:|---:|
| 0.1 | 5/6 | 2/6 | 6/6 |
| 0.25 | 6/6 | 2/6 | 6/6 |
| 0.5 | 6/6 | 6/6 | 6/6 |
| 0.75 | 5/6 | 5/6 | 6/6 |
| 0.9 | 5/6 | 5/6 | 5/6 |

## Ablations @~127K, depth 0.5, first 6 prompts

| routed columns | exact |
|---:|---:|
| 64 | 0/6 |
| 256 | 6/6 |
| 512 | 6/6 |
| 1024 | 5/6 |
| 2048 | 4/6 |
| 3072 | 3/6 |

| exact tail | exact |
|---:|---:|
| 0 | 2/6 |
| 64 | 4/6 |
| 256 | 6/6 |

## All arms @~127K, first 6 prompts (tuning set)

| run | arm | exact | prefill s |
|---|---|---:|---:|
| b1_dense | Dense FlashAttention | 6/6 | 16.33 |
| b1_exp19 | exp19 Bigger Bird (prev.) | 0/6 | 4.33 |
| b1_vs_s64_w4k_v2k | sink64+win4096+vert2048 | 2/6 | 5.31 |
| b2_v2k_w4k_max | sink64+win4096+vert2048 (max-pool) | 1/6 | 5.36 |
| b2_v2k_w4k_r256 | sink64+win4096+vert2048 (route q=256) | 0/6 | 5.77 |
| b2_v4k_w4k | sink64+win4096+vert4096 | 1/6 | 5.80 |
| b2_v4k_w4k_max | sink64+win4096+vert4096 (max-pool) | 0/6 | 5.86 |
| b2_v4k_w4k_max_r16 | sink64+win4096+vert4096 (max-pool) (route q=16) | 0/6 | 5.61 |
| b2_v8k_w8k | sink64+win8192+vert8192 | 0/6 | 7.56 |
| b3_v0_w4k_t64 | sink64+win4096+vert64+tail64 | 0/6 | 5.30 |
| b3_v1k_w2k_t64 | sink64+win2048+vert1024+tail64 | 1/6 | 5.16 |
| b3_v2k_w4k_t256 | sink64+win4096+vert2048+tail256 | 4/6 | 6.55 |
| b3_v2k_w4k_t64 | sink64+win4096+vert2048+tail64 | 3/6 | 5.82 |
| b3_v4k_w4k_t64 | sink64+win4096+vert4096+tail64 | 0/6 | 6.35 |
| b3_v8k_w8k_t64 | sink64+win8192+vert8192+tail64 | 0/6 | 8.28 |
| b4_v1k_w4k_t256 | sink64+win4096+vert1024+tail256 | 5/6 | 6.25 |
| b4_v2k_w2k_t256 | sink64+win2048+vert2048+tail256 | 1/6 | 6.09 |
| b4_v2k_w4k_t256_r256 | sink64+win4096+vert2048+tail256 (route q=256) | 1/6 | 6.92 |
| b4_v2k_w4k_t512 | sink64+win4096+vert2048+tail512 | 4/6 | 7.48 |
| b4_v2k_w8k_t256 | sink64+win8192+vert2048+tail256 | 4/6 | 7.24 |
| b4_v3k_w4k_t256 | sink64+win4096+vert3072+tail256 | 3/6 | 6.82 |
| b5_last_v1k_t512 | sink64+win4096+vert1024+tail512 | 5/6 | 7.18 |
| b5_last_v512_t256 | sink64+win4096+vert512+tail256 | 6/6 | 6.09 |
| b5_spread_v1k_t256 | sink64+win4096+vert1024+tail256 (spread) (route q=256) | 1/6 | 6.76 |
| b5_spread_v2k_t256 | sink64+win4096+vert2048+tail256 (spread) (route q=256) | 1/6 | 6.91 |
| b5_spread_v4k_t256 | sink64+win4096+vert4096+tail256 (spread) (route q=256) | 1/6 | 7.45 |
| b5_spread_v512_t256 | sink64+win4096+vert512+tail256 (spread) (route q=256) | 3/6 | 6.43 |
| b6_abl_v256_t256 | sink64+win4096+vert256+tail256 | 6/6 | 6.04 |
| b6_abl_v512_t0 | sink64+win4096+vert512 | 2/6 | 4.95 |
| b6_abl_v512_t64 | sink64+win4096+vert512+tail64 | 4/6 | 5.45 |
| b6_abl_v64_t256 | sink64+win4096+vert64+tail256 | 0/6 | 6.03 |
| b6_scale_best | sink64+win4096+vert512+tail256 | 6/6 | 6.12 |
| b6_scale_dense | Dense FlashAttention | 6/6 | 16.22 |
