#!/usr/bin/env bash
# Batch 10: Triton tile/warp tuning for the final config @~127K (3 prompts, warmed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
job() {  # tag extra-json
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '{\"sink\":64,\"window\":16384,\"vertical\":512,\"exact_tail\":256,$2}' --bytes 512000 --n 3 --tag $1" \
    > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
job b10_m64_n64_w4    '"block_m":64,"block_n":64,"num_warps":4'
job b10_m128_n64_w8   '"block_m":128,"block_n":64,"num_warps":8'
job b10_m128_n128_w8  '"block_m":128,"block_n":128,"num_warps":8'
job b10_m64_n128_w4   '"block_m":64,"block_n":128,"num_warps":4'
job b10_m128_n64_w4s3 '"block_m":128,"block_n":64,"num_warps":4,"num_stages":3'
wait
