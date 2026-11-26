#!/usr/bin/env bash
# Week-3 end-to-end run. One model, one GPU, no training.
#
#   export MODEL=/data/models/Qwen2.5-Coder-7B-Instruct
#   bash run_all.sh
#
# Optional knobs:
#   PY=python3.11   TAG=mytag   N=40 (gate items)   SKIP_CURVE=1   SKIP_4K=1
# Everything is logged under results/$TAG/.
set -euo pipefail

: "${MODEL:?set MODEL to a local checkpoint path or a Hugging Face id}"
PY=${PY:-python}
TAG=${TAG:-$(date +%Y%m%d_%H%M)}
N=${N:-40}
OUT="results/$TAG"
mkdir -p "$OUT"

run() {  # run <logname> <command...>
  local name=$1; shift
  echo
  echo "=== $name ==="
  echo "+ $*"
  "$@" 2>&1 | tee "$OUT/$name.log"
}

cd "$(dirname "$0")"

# 0. environment, tests, and the no-weights preflight
run 00_lab_check   "$PY" lab_check.py --model "$MODEL"
run 01_unit        "$PY" -m unittest -v test_week3
run 02_preflight   "$PY" w3_preflight.py --lengths 256 1024 --make-length 4096 --sample 8 --layers 28 36

# 1. gate: if the readout cannot copy a visible digit, nothing downstream counts
if ! "$PY" w3_gate.py --model "$MODEL" --style completion --limit "$N" --out "$OUT/gate.json" \
      2>&1 | tee "$OUT/03_gate.log"; then
  echo
  echo "GATE FAILED. Fix the prompt/readout first (see LAB_SETUP_ZH.md, Troubleshooting)."
  echo "Log: $OUT/03_gate.log"
  exit 2
fi

# 2. ceiling on the original 3-hop data (bytes x family)
if [[ "${SKIP_CURVE:-0}" != "1" ]]; then
  run 04_curve "$PY" w3_curve.py --model "$MODEL" --lengths 96 160 256 512 --per 20 \
      --out "$OUT/curve.csv"
fi

# 3. depth x distance: where does a ceiling actually exist?
#    The committed data/needle_test.jsonl was generated before the calibration fix
#    (6822708); regenerating it would produce different lengths and silently change
#    table 3, so the committed file is used as-is unless it is missing.
if [[ -f data/needle_test.jsonl ]]; then
  echo
  echo "=== 05_needle_gen ==="
  echo "using the committed data/needle_test.jsonl (generated at commit 5ba9f1a)"
else
  run 05_needle_gen "$PY" w3_needle.py --distances 20 100 400 1600 --hops 1 2 3 --per 20
fi
run 06_needle_full "$PY" w3_run.py --model "$MODEL" --data-dir data --split needle_test \
    --lengths 0 --group-by family_id --modes full --no-reach --out "$OUT/needle_full"

# 3b. regenerate needle_h1 as the 250-program, five-distance set used by tables 4, 8 and 10.
#     This overwrites data/needle_h1.jsonl written by step 05 (80 programs, four distances).
run 06b_needle_long "$PY" w3_needle.py --distances 20 100 400 1600 4000 --hops 1 \
    --per 50 --split needle_long

# 4. visibility comparison on the 1-hop set (the one with a real ceiling)
run 07_vis_h1 "$PY" w3_run.py --model "$MODEL" --data-dir data --split needle_h1 \
    --lengths 0 --group-by family_id --no-reach \
    --modes full,window=64,window=256,gold_union=64,block=4,random=64 --out "$OUT/vis_h1"

# 4b. deployable fixed anchors vs the completeness control (Table 8)
run 07b_vis_localglobal "$PY" w3_run.py --model "$MODEL" --data-dir data --split needle_h1 \
    --lengths 0 --group-by family_id --no-reach \
    --modes full,local_global=64:first:4,local_global=64:linestart:1,local_global=64:linestart:4 \
    --out "$OUT/vis_localglobal"

# 4c. budget-matched control (Table 10): window=100 / random=100 vs gold_union=64
run 07c_vis_matched "$PY" w3_run.py --model "$MODEL" --data-dir data --split needle_h1 \
    --lengths 0 --group-by family_id --no-reach \
    --modes full,window=64,window=100,random=100,gold_union=64 --out "$OUT/vis_matched"

# 5. evidence dependence on the original harder set
run 08_counterfactual "$PY" w3_run.py --model "$MODEL" --split test --lengths 256 \
    --style completion --ablate gold-value --modes full --out "$OUT/cf"
run 09_drop "$PY" w3_run.py --model "$MODEL" --split test --lengths 256 \
    --style completion --ablate gold-drop --modes full --out "$OUT/drop"

# 6. 4 KB programs, where a 64-token window finally cuts the evidence
if [[ "${SKIP_4K:-0}" != "1" ]]; then
  run 10_vis_4k "$PY" w3_run.py --model "$MODEL" --data-dir preflight --split preview_l4096 \
      --lengths 4096 --style completion --limit 60 \
      --modes full,window=64,gold_union=64 --out "$OUT/vis_4k"
fi

# 7. one-screen summary
echo
echo "=== summary ($OUT) ==="
for f in "$OUT"/gate.json "$OUT"/needle_full/summary.csv "$OUT"/vis_h1/summary.csv \
         "$OUT"/cf/summary.csv "$OUT"/drop/summary.csv "$OUT"/vis_4k/summary.csv; do
  [[ -f $f ]] || continue
  echo
  echo "--- $(basename "$(dirname "$f")")/$(basename "$f") ---"
  cat "$f"
done

echo
echo "Bundle for the write-up:  tar czf week3_$TAG.tgz $OUT preflight/preflight.md data/needle_*.jsonl"
