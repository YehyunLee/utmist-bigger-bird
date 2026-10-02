#!/usr/bin/env bash
# Batch 1: baselines + diagnosis, all in parallel on separate H100s (~$10-15 total).
# Results: modal volume ls bb-vol results ; modal volume get bb-vol results/<tag>.json .
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { modal run --detach modal_rd/app.py --module "$1" --args "$2" > "modal_rd/logs/$3.log" 2>&1 & }
mkdir -p modal_rd/logs

# Fresh dense baseline (KV cache) at ~65K and ~127K on the CCDB-matched prompts.
[[ "${SKIP_DENSE:-}" ]] || launch modal_rd.eval_niah "--arm dense --bytes 261000,512000 --n 6 --tag b1_dense" b1_dense
# Current exp19 default (last_query, no sink) on the same prompts and protocol.
launch modal_rd.eval_niah "--arm exp19 --bytes 261000,512000 --n 6 --tag b1_exp19" b1_exp19
# New all-sparse candidate: sink64 + local4096 + per-head vertical2048.
launch modal_rd.eval_niah "--arm vs --attn '{\"sink\":64,\"window\":4096,\"vertical\":2048}' --bytes 261000,512000 --n 6 --tag b1_vs_s64_w4k_v2k" b1_vs_s64_w4k_v2k
# Per-layer dense attention-mass recall of candidate patterns (2 prompts @127K, 1 @65K).
launch modal_rd.recall "--bytes 512000 --n 2 --tag b1_recall_127k" b1_recall_127k
launch modal_rd.recall "--bytes 261000 --n 1 --tag b1_recall_65k" b1_recall_65k
wait
echo "launched; follow with: tail -f modal_rd/logs/*.log"
