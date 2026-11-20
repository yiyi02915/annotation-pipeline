# Local preference-training prototype

This prototype adapts localized pairwise preference learning to duplex annotations.
It trains the text decoder of an existing DuplexJev-4B-Para checkpoint; the audio
encoder is not trained. It is not a reproduction of SuperFace's face model.

Copy these scripts into a separate experiment directory containing `train.jsonl`,
`validation.jsonl`, `test.jsonl`, and `regression.jsonl`. Each row has `id`,
`conversation_id`, `task` (`gap` or `overlap`), `input`, and an independently
established `label`. Split by original conversation before constructing pairs.
Reference labels are targets only and are not inserted into the model evidence.

`train.py` currently uses the existing server checkpoint path near its model load.
Change it for another installation. It requires torch, transformers, peft and
DuplexJev in the same working environment. Run on a dedicated GPU:

```bash
CUDA_VISIBLE_DEVICES=2 python train.py > train.log 2>&1
```

Stages: baseline evaluation, LoRA preference judge training on AB/BA comparisons,
preference filtering with independent seed fallback, balanced policy SFT, one DPO
round against the frozen SFT reference probabilities, and held-out evaluation.
The reference margins are computed before DPO updates. Validation macro recall
selects a checkpoint; test and regression labels are never used for gradients.

This is a small seed-data experiment. Repeated/balanced examples are not independent
new samples. Inspect per-class recall: overall accuracy can conceal collapse of
interruptions or turn transitions. A consistent judge is not necessarily correct.
The same-model judge and policy can share errors, and seed fallback means not all
DPO pairs were independently discovered by the judge.

Outputs are `judge_lora`, `policy_sft_lora`, `policy_dpo_lora`, `metrics.json`,
`judge_metrics.json`, training losses and provenance/configuration. The current
research data are model-adjudicated references, not human gold. No raw recordings,
reference datasets, credentials or pretrained model weights belong in this folder.

Reload a policy adapter:

```bash
CUDA_VISIBLE_DEVICES=2 python infer.py --base /path/to/DuplexJev-4B-Para \
  --adapter policy_dpo_lora --input test.jsonl --output predictions.json
```

The output preserves sample IDs and maps the uniform single-token codes back to
normal labels. `NEITHER` is a candidate filter for overlap, never a third gap label.
