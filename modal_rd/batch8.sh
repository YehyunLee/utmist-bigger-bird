#!/usr/bin/env bash
# Batch 8: wider local windows for shallow needles @~127K.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
job() {  # tag json
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes 512000 --n 6 --depths 0.1,0.25,0.5 --warm first --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
job b8_w8k_v512   '{"sink":64,"window":8192,"vertical":512,"exact_tail":256}'
job b8_w16k_v512  '{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
job b8_w8k_v1k    '{"sink":64,"window":8192,"vertical":1024,"exact_tail":256}'
job b8_w4k_v512_s256 '{"sink":256,"window":4096,"vertical":512,"exact_tail":256}'
wait
