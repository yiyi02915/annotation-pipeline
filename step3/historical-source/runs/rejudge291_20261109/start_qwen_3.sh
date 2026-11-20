#!/usr/bin/env bash
set -euo pipefail
RUNTIME=/data/L202500493/duplex_postprocess/runtime/glm
ROOT=/data/L202500493/duplex_postprocess/runs/finegrained_pilot_20261109
export CUDA_VISIBLE_DEVICES=3 VLLM_PLUGINS='' VLLM_USE_RUST_FRONTEND=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTHONPATH="$RUNTIME/runtime_hooks"
export CUDA_HOME="$RUNTIME/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
exec "$RUNTIME/venv/bin/python" -m vllm.entrypoints.openai.api_server \
 --model /data2/202610/Qwen3.8-27B --served-model-name qwen38-27b \
 --host 127.0.0.1 --port 18355 --tensor-parallel-size 1 \
 --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 4096 \
 --gpu-memory-utilization 0.90 --enforce-eager \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}'
