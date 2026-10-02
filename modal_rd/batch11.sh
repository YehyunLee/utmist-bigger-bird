#!/usr/bin/env bash
# Batch 11: harder RULER NIAH variants @~127K tokens, dense vs final config (untuned on these tasks).
# niah_multikey_2 is excluded: its distractor needles reuse the query key with other values (ill-posed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
FINAL='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
job() {  # tag args
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah --args "$2 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
for spec in "niah_single_2:630000:16" "niah_single_3:630000:40" "niah_multikey_1:630000:16" \
            "niah_multikey_3:390000:40" "niah_multivalue:630000:64" "niah_multiquery:630000:64"; do
  IFS=: read -r task bytes newtok <<< "$spec"
  common="--task $task --bytes $bytes --n 12 --max-new $newtok --warm first"
  job "b11_${task}_dense" "--arm dense $common"
  job "b11_${task}_final" "--arm vs --attn '$FINAL' $common"
done
wait
