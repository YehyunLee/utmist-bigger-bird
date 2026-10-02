#!/usr/bin/env bash
# Batch 14: question moved to the start of the prompt @~127K (same 6 prompts as the question-at-end screens).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
FINAL='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
  --args "--arm vs --attn '$FINAL' --bytes 512000 --n 6 --question-first --tag b14_qfirst_final" \
  > modal_rd/logs/b14_qfirst_final.log 2>&1 &
sleep 3
modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
  --args "--arm dense --bytes 512000 --n 6 --question-first --tag b14_qfirst_dense" \
  > modal_rd/logs/b14_qfirst_dense.log 2>&1 &
wait
# Confound check: question ~500 tokens in, outside the 64 always-visible start tokens.
modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
  --args "--arm vs --attn '$FINAL' --bytes 512000 --n 6 --question-first --question-offset 2000 --tag b14_qoff_final" \
  > modal_rd/logs/b14_qoff_final.log 2>&1 &
sleep 3
modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
  --args "--arm dense --bytes 512000 --n 6 --question-first --question-offset 2000 --tag b14_qoff_dense" \
  > modal_rd/logs/b14_qoff_dense.log 2>&1 &
wait
