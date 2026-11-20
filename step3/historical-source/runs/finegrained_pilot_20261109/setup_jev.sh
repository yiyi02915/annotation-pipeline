#!/usr/bin/env bash
set -euo pipefail
BASE=/data/L202500293/envs/qwen3-asr/bin/python
DEST=/data/L202500493/duplex_postprocess/runtime/jev
if [ ! -f "$DEST/pyvenv.cfg" ]; then "$BASE" -m venv --system-site-packages "$DEST"; fi
"$DEST/bin/python" -m pip install --no-deps duplexjev==0.7.0 peft==0.17.1
"$DEST/bin/python" -c 'import torch,transformers,duplexjev,peft; print(torch.__version__,transformers.__version__,peft.__version__); print("JEV_ENV_READY")'
