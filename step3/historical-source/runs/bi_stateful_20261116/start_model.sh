#!/bin/bash
cd /data/L202500493/duplex_postprocess/runs/bi_stateful_20261116
export VLLM_PLUGINS=""
export VLLM_USE_RUST_FRONTEND=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/L202500493/duplex_postprocess/runtime/glm/runtime_hooks
export CUDA_HOME=/data/L202500493/duplex_postprocess/runtime/glm/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
nohup env CUDA_VISIBLE_DEVICES=3 /data/L202500493/duplex_postprocess/runtime/glm/venv/bin/python -m vllm.entrypoints.openai.api_server --model /data2/202610/Qwen3.8-27B --served-model-name qwen38-27b --host 127.0.0.1 --port 18771 --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 2048 --gpu-memory-utilization .92 --enforce-eager --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' >model.log 2>&1 </dev/null &
echo $! >model.pid
cat model.pid
