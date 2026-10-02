#!/usr/bin/env bash
# Batch 7: fix shallow-needle failures by limiting which queries see routed columns.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
job() {  # tag json
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes 512000 --n 6 --depths 0.1,0.25,0.5 --warm first --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
B='"sink":64,"window":4096,"exact_tail":256'
job b7_v512_scope8k  "{$B,\"vertical\":512,\"vertical_scope\":8192}"
job b7_v512_scope32k "{$B,\"vertical\":512,\"vertical_scope\":32768}"
job b7_v2k_scope8k   "{$B,\"vertical\":2048,\"vertical_scope\":8192}"
job b7_v256          "{$B,\"vertical\":256}"
job b7_v512_t1024s   "{$B,\"vertical\":512,\"vertical_scope\":4096}"
wait
