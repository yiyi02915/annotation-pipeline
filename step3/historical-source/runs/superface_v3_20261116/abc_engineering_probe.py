"""Exercise A/B/C checkpoint code on old training rows only, before full experiment."""
import json
from pathlib import Path
import torch
import train_abc_multimodal as runner
from multimodal_model import Model

ROOT=Path(__file__).resolve().parent


def main():
    rows=json.loads((ROOT/'smoke32_inputs.json').read_text())
    runner.ROOT=ROOT/'engineering_probe'
    model=Model();results=[]
    for arm in ['A','B','C']:
        for task in ['gap','overlap']:
            subset=[r for r in rows if r['task']==task]
            torch.manual_seed(20261116)
            result=runner.fit_task(model,arm,task,subset[:4],subset[4:6])
            state_path=Path(json.loads((runner.ROOT/'abc'/arm/task/'latest.json').read_text())['checkpoint'])/'state.pt'
            state=torch.load(state_path,map_location='cpu',weights_only=False)
            assert state['optimizer']['state'] and state['updates']>0
            if arm=='C': assert result['reference_fixed_to']
            results.append(dict(arm=arm,task=task,updates=state['updates'],optimizer_saved=True))
    (ROOT/'checkpoint_abc_engineering.json').write_text(json.dumps(dict(status='passed',checks=results,scope='old training rows only; code-path verification, not model-quality comparison'),indent=2))


if __name__=='__main__':main()
