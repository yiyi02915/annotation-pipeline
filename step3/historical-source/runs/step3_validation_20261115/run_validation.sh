#!/bin/bash
set -u
cd /data/L202500493/duplex_postprocess/runs/step3_validation_20261115
PY=/data/L202500493/duplex_postprocess/runtime/jev/bin/python
DATA=/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd
export PYTHONDONTWRITEBYTECODE=1
$PY large_precheck.py > sample.log 2>&1
$PY step3/pipeline.py --seamless-root "$DATA" --pairs random2000.jsonl --output precheck2000 --prepare-only > precheck.log 2>&1
echo $? > precheck.exit
CUDA_VISIBLE_DEVICES=3 $PY step3/pipeline.py --seamless-root "$DATA" --pairs pairs.jsonl --output run --jev-checkpoint /data/L202500493/duplex_postprocess/models/DuplexJev-32B-Para > run.log 2>&1
echo $? > run.exit
$PY score.py > score.log 2>&1
echo $? > score.exit
