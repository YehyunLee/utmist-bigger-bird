#!/usr/bin/env bash
# Batch 13: essay-haystack sweep @~127K on prompts 12-23 (prompts 0-11 stay untouched for reporting).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
job() {  # tag json
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --task niah_single_2 --bytes 630000 --n 12 --start 12 --max-new 16 --warm first --tag $1" \
    > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
job b13_final          '{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
job b13_v1k            '{"sink":64,"window":16384,"vertical":1024,"exact_tail":256}'
job b13_v2k            '{"sink":64,"window":16384,"vertical":2048,"exact_tail":256}'
job b13_w32k           '{"sink":64,"window":32768,"vertical":512,"exact_tail":256}'
job b13_t1024          '{"sink":64,"window":16384,"vertical":512,"exact_tail":1024}'
job b13_r16            '{"sink":64,"window":16384,"vertical":512,"exact_tail":256,"n_route":16}'
modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
  --args "--arm dense --task niah_single_2 --bytes 630000 --n 12 --start 12 --max-new 16 --warm first --tag b13_dense" \
  > modal_rd/logs/b13_dense.log 2>&1 &
wait
