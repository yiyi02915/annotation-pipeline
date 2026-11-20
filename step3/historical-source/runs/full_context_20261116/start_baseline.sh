#!/bin/bash
cd /data/L202500493/duplex_postprocess/runs/full_context_20261116
nohup env PYTHONUNBUFFERED=1 /data/L202500493/duplex_postprocess/runtime/jev/bin/python current_baseline.py >baseline_current.log 2>&1 </dev/null &
echo $! >baseline_current.pid
cat baseline_current.pid
