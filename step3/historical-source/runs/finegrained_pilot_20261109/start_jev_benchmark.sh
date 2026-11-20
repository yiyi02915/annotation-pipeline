#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/L202500493/duplex_postprocess/runs/finegrained_pilot_20261109
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/data/L202500493/duplex_postprocess/runtime/jev/bin/python
"$PY" -u "$ROOT/run_jev.py" --limit 2
"$PY" -u "$ROOT/run_jev.py" --batch-size 4
