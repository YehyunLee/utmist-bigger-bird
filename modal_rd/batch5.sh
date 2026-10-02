#!/usr/bin/env bash
# Batch 5: globally-spread vertical routing + exact question tail at ~127K.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
arm() {  # tag json [bytes]
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes ${3:-512000} --n 6 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
S='"sink":64,"window":4096,"route_queries":"spread","n_route":256'
arm b5_spread_v1k_t256 "{$S,\"vertical\":1024,\"exact_tail\":256}"
arm b5_spread_v2k_t256 "{$S,\"vertical\":2048,\"exact_tail\":256}"
arm b5_spread_v4k_t256 "{$S,\"vertical\":4096,\"exact_tail\":256}"
arm b5_spread_v512_t256 "{$S,\"vertical\":512,\"exact_tail\":256}"
arm b5_last_v1k_t512 '{"sink":64,"window":4096,"vertical":1024,"exact_tail":512}'
arm b5_last_v512_t256 '{"sink":64,"window":4096,"vertical":512,"exact_tail":256}'
wait
