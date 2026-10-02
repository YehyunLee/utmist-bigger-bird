#!/usr/bin/env bash
# Batch 12: harder RULER NIAH variants @~32K and ~65K tokens (dense is near-ceiling there), dense vs final.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
FINAL='{"sink":64,"window":16384,"vertical":512,"exact_tail":256}'
job() {  # tag args
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah --args "$2 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
for spec in "niah_single_2:16" "niah_multikey_1:16" "niah_multivalue:64" "niah_multiquery:64"; do
  IFS=: read -r task newtok <<< "$spec"
  common="--task $task --bytes 161000,322000 --n 12 --max-new $newtok --warm first"
  job "b12_${task}_dense" "--arm dense $common"
  job "b12_${task}_final" "--arm vs --attn '$FINAL' $common"
done
wait
