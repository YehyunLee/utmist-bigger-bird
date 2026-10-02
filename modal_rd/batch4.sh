#!/usr/bin/env bash
# Batch 4: sweep around the best arm (v2k w4k exact_tail256) at ~127K.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
arm() {  # tag json [bytes]
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes ${3:-512000} --n 6 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
arm b4_v2k_w4k_t512   '{"sink":64,"window":4096,"vertical":2048,"exact_tail":512}'
arm b4_v2k_w4k_t1024  '{"sink":64,"window":4096,"vertical":2048,"exact_tail":1024}'
arm b4_v1k_w4k_t256   '{"sink":64,"window":4096,"vertical":1024,"exact_tail":256}'
arm b4_v3k_w4k_t256   '{"sink":64,"window":4096,"vertical":3072,"exact_tail":256}'
arm b4_v2k_w8k_t256   '{"sink":64,"window":8192,"vertical":2048,"exact_tail":256}'
arm b4_v2k_w2k_t256   '{"sink":64,"window":2048,"vertical":2048,"exact_tail":256}'
arm b4_v2k_w4k_t256_r256 '{"sink":64,"window":4096,"vertical":2048,"exact_tail":256,"n_route":256}'
arm b4_v512_w4k_t1024 '{"sink":64,"window":4096,"vertical":512,"exact_tail":1024}'
wait
