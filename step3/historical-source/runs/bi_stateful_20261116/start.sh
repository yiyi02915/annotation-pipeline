#!/bin/bash
cd /data/L202500493/duplex_postprocess/runs/bi_stateful_20261116
nohup env PYTHONUNBUFFERED=1 /data/L202500493/duplex_postprocess/runtime/jev/bin/python run.py >run.log 2>&1 </dev/null &
echo $! >run.pid
cat run.pid
