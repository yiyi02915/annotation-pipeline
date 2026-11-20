#!/bin/bash
cd /data/L202500493/duplex_postprocess/runs/full_context_20261116
nohup env PYTHONUNBUFFERED=1 /data/L202500493/duplex_postprocess/runtime/jev/bin/python anchored_bi.py >anchored_bi.log 2>&1 </dev/null &
echo $! >anchored_bi.pid
cat anchored_bi.pid
