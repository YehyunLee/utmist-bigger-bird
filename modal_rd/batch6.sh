#!/usr/bin/env bash
# Batch 6: confirmation (30 prompts @65K/127K), context scaling, depth sweep, ablations.
# Best arm from batches 1-5 (tuned on the first 6 prompts @127K): BEST below.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
BEST='{"sink":64,"window":4096,"vertical":512,"exact_tail":256}'
job() {  # tag args
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah --args "$2 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
# Confirmation: examples 0-5 tuned the arm; report 6-29 separately.
job b6_confirm_dense "--arm dense --bytes 261000,512000 --n 30"
job b6_confirm_best  "--arm vs --attn '$BEST' --bytes 261000,512000 --n 30"
# Context scaling (~8K..127K tokens), timed after warming every prompt.
job b6_scale_dense   "--arm dense --bytes 32000,64000,130000,261000,512000 --n 6"
job b6_scale_best    "--arm vs --attn '$BEST' --bytes 32000,64000,130000,261000,512000 --n 6"
# Needle depth sweep @127K (accuracy only).
job b6_depth_dense   "--arm dense --bytes 512000 --n 6 --depths 0.1,0.25,0.75,0.9 --warm first"
job b6_depth_best    "--arm vs --attn '$BEST' --bytes 512000 --n 6 --depths 0.1,0.25,0.75,0.9 --warm first"
# Ablations @127K.
job b6_abl_v64_t256   "--arm vs --attn '{\"sink\":64,\"window\":4096,\"vertical\":64,\"exact_tail\":256}' --bytes 512000 --n 6"
job b6_abl_v512_t0    "--arm vs --attn '{\"sink\":64,\"window\":4096,\"vertical\":512}' --bytes 512000 --n 6"
job b6_abl_v512_t64   "--arm vs --attn '{\"sink\":64,\"window\":4096,\"vertical\":512,\"exact_tail\":64}' --bytes 512000 --n 6"
job b6_abl_v256_t256  "--arm vs --attn '{\"sink\":64,\"window\":4096,\"vertical\":256,\"exact_tail\":256}' --bytes 512000 --n 6"
wait
