"""32 real dual-track inputs: gradients, frozen audio, save/reload and modalities."""
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from multimodal_model import Model

ROOT=Path(__file__).resolve().parent


def save(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)


def main():
    random.seed(20261116);torch.manual_seed(20261116)
    rows=json.loads((ROOT/'revised_train.json').read_text())
    # This is an engineering check, not an accuracy evaluation or a selected deployment model.
    rows=[r for r in rows if r['audio_ready'] and r['audio_source']['kind']=='synchronized_dual_track_mix']
    selected=[]
    for task in ['gap','overlap']:
        pool=[r for r in rows if r['task']==task]
        pool.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        assert len(pool)>=16
        selected+=pool[:16]
    save('smoke32_inputs.json',selected)
    model=Model();results=[];start=time.monotonic()
    for task in ['gap','overlap']:
        model.adapter();model.model.language_model.train()
        optimizer=torch.optim.AdamW(model.trainable(),lr=1e-4)
        subset=[r for r in selected if r['task']==task]
        gradient_seen=False
        for i,row in enumerate(subset,1):
            optimizer.zero_grad(set_to_none=True)
            calls=model.audio_forward_calls
            loss=model.loss(row)
            assert torch.isfinite(loss) and model.audio_forward_calls>calls
            loss.backward();model.assert_frozen()
            grad=float(torch.nn.utils.clip_grad_norm_(model.trainable(),1.0))
            assert grad>0 and torch.isfinite(torch.tensor(grad))
            gradient_seen=True;optimizer.step()
            results.append(dict(id=row['id'],task=task,step=i,loss=float(loss.detach()),gradient_norm=grad,audio_forward=True))
            save('smoke32_progress.json',dict(completed=len(results),total=32,results=results))
            if i%8==0:
                ckpt=ROOT/'smoke32'/task/f'step_{i:03d}';ckpt.mkdir(parents=True,exist_ok=True)
                model.model.language_model.save_pretrained(ckpt)
                torch.save(dict(optimizer=optimizer.state_dict(),step=i,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),ckpt/'state.pt')
            print('SMOKE',task,i,float(loss.detach()),flush=True)
        assert gradient_seen
        model.model.language_model.eval()
        with torch.no_grad():
            before=model.scores(subset[0]).cpu()
            text=model.scores(subset[0],audio=False).cpu()
        model.adapter(ckpt);model.model.language_model.eval()
        with torch.no_grad(): after=model.scores(subset[0]).cpu()
        delta=float((after-before).abs().max())
        assert delta<.05,('save_reload_mismatch',delta)
        save(f'smoke32/{task}/verification.json',dict(reload_max_logit_difference=delta,audio_vs_text_scores_difference=float((before-text).abs().max()),frozen_audio_parameters=True,nonzero_lora_gradients=True,inputs=16))
    save('checkpoint_03_smoke32.json',dict(status='passed',inputs=32,audio_forward_calls=model.audio_forward_calls,wall_seconds=time.monotonic()-start,tasks=['gap','overlap'],scope='engineering viability only; not accuracy evidence',checkpoints=4))


if __name__=='__main__': main()
