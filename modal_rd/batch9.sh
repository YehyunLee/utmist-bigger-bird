#!/usr/bin/env bash
# Batch 9: final config (tuned on prompts 0-5 at depths 0.1/0.25/0.5 @~127K) -> confirmation + charts.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
FINAL='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
job() {  # tag args
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah --args "$2 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
job b9_confirm_final       "--arm vs --attn '$FINAL' --bytes 261000,512000 --n 30"
job b9_confirm_final_d01   "--arm vs --attn '$FINAL' --bytes 512000 --n 30 --depths 0.1"
job b9_confirm_dense_d01   "--arm dense --bytes 512000 --n 30 --depths 0.1"
job b9_scale_final         "--arm vs --attn '$FINAL' --bytes 32000,64000,130000,261000,512000 --n 6"
job b9_depth_final         "--arm vs --attn '$FINAL' --bytes 512000 --n 6 --depths 0.1,0.25,0.75,0.9"
wait
