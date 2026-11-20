#!/bin/bash
cd /data/L202500493/duplex_postprocess/runs/full_context_20261116
nohup env PYTHONUNBUFFERED=1 /data/L202500493/duplex_postprocess/runtime/jev/bin/python repair_context.py >repair_context.log 2>&1 </dev/null &
echo $! >repair_context.pid
cat repair_context.pid
