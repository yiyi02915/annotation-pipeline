"""Reload a trained semantic-policy adapter and retain original sample IDs."""
import argparse,json
from pathlib import Path
from duplexjev import Decider
from peft import PeftModel
import model_format as fmt

p=argparse.ArgumentParser()
p.add_argument('--base',default='/data/L202500493/duplex_postprocess/models/DuplexJev-4B-Para')
p.add_argument('--adapter',required=True)
p.add_argument('--input',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--limit',type=int,default=0)
args=p.parse_args()
engine=Decider.from_pretrained(args.base,device='cuda:0',attn_implementation='sdpa')
fmt.tokenizer=engine.tok
assert all(len(fmt.tokenizer(code,add_special_tokens=False)['input_ids'])==1 for code in fmt.CODES.values())
model=PeftModel.from_pretrained(engine.model.language_model,args.adapter).eval()
rows=[json.loads(line) for line in args.input.read_text().splitlines()]
if args.limit:rows=rows[:args.limit]
results=[]
for row in rows:
    choices=[fmt.CODES[label] for label in fmt.OPTIONS[row['task']]]
    code,scores=fmt.choose(model,fmt.prompt(row),choices)
    label=next(label for label in fmt.OPTIONS[row['task']] if fmt.CODES[label]==code)
    results.append(dict(id=row['id'],task=row['task'],predicted=label,scores=scores))
args.output.write_text(json.dumps(results,indent=2))
print('RELOAD_INFERENCE_COMPLETE',flush=True)
