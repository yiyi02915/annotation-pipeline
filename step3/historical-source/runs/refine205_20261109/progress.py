import ast,json
from pathlib import Path
D=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
tree=ast.parse((D/'clarify_facts.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='consistent');scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'consistent','exec'),scope)
raw={x['id']:x for k in range(2) for x in read(D/f'qwen_{k}.jsonl')};raw.update({x['id']:x for x in read(D/'qwen_repair.jsonl') if x['status']=='ok'});needed={k for k,v in raw.items() if v['status']=='ok' and v.get('finish_reason')=='stop' and not scope['consistent'](v['judgment'])};done={x['id'] for x in read(D/'qwen_fact_clarification.jsonl')}
print({'qwen':len(raw),'clarification_needed':len(needed),'clarified':len(done),'clarification_pending':len(needed-done),'clarification_complete':'CLARIFICATION_COMPLETE' in (D/'clarify_facts.log').read_text()})
