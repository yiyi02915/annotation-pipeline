import json,argparse,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from llm_client import Client,BOUNDARY_SCHEMA
from pipeline_core import validate_boundary

def adjudicate(root,url='http://127.0.0.1:18351/v1',model='glm-judge'):
    root=Path(root);client=Client(root,url,model)
    data=json.loads((root/'disputed_inputs.json').read_text(encoding='utf8'))
    standard=(root/'boundary_prompt.txt').read_text(encoding='utf8')
    jobs=[(c,role) for c in data for role in ['independent_annotator','independent_auditor']]
    def one(job):
        c,role=job
        instruction='Identify action and response target on both sides.' if role.endswith('annotator') else 'Audit action continuity independently, including quoted questions and overlap chronology.'
        payload={k:c[k] for k in ['left','right','gap_seconds','context']}
        raw=client.call(standard+'\n'+instruction,payload,BOUNDARY_SCHEMA,c['candidate_id']+':'+role)
        decision,error=validate_boundary(raw['parsed'])
        print(c['previous_candidate_id'],role,decision,error,flush=True)
        return dict(candidate_id=c['candidate_id'],role=role,decision=decision or 'PENDING_AUTOMATIC',error=error,result=raw['parsed'],request_hash=raw['request_hash'])
    with ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(one,jobs))
    if not any(r['error'] is None for r in responses):
        (root/'adjudication_failures.json').write_text(json.dumps(responses,ensure_ascii=False,indent=2),encoding='utf8')
        raise RuntimeError('Independent model returned no valid annotation; inspect model_calls.jsonl before continuing')
    final=[]
    for c in data:
        pair=[r for r in responses if r['candidate_id']==c['candidate_id']]
        same=pair[0]['decision']==pair[1]['decision'] and pair[0]['decision']!='PENDING_AUTOMATIC'
        decision=pair[0]['decision'] if same else 'PENDING_AUTOMATIC'
        # One explicit adjudication of disagreement, not repeated sampling until a preferred label wins.
        arbitration=None
        if not same:
            payload={k:c[k] for k in ['left','right','gap_seconds','context']}
            payload['conflicting_analyses']=[r['result'] for r in pair]
            rr=client.call(standard+'\nResolve the disagreement using specific evidence; do not majority-vote. Use F if still ambiguous.',payload,BOUNDARY_SCHEMA,c['candidate_id']+':arbiter')
            decision,error=validate_boundary(rr['parsed']);decision=decision or 'PENDING_AUTOMATIC'
            arbitration=dict(result=rr['parsed'],error=error,request_hash=rr['request_hash'])
        final.append(dict(candidate_id=c['candidate_id'],previous_candidate_id=c['previous_candidate_id'],decision=decision,origin='independent_glm_adjudication',responses=pair,arbitration=arbitration,human_required=False))
    (root/'adjudicated_boundaries.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8')
    manifest=dict(model=model,input_sha256=hashlib.sha256((root/'disputed_inputs.json').read_bytes()).hexdigest(),prompt_sha256=hashlib.sha256((root/'boundary_prompt.txt').read_bytes()).hexdigest(),results_sha256=hashlib.sha256((root/'adjudicated_boundaries.json').read_bytes()).hexdigest())
    (root/'adjudication_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return final

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18351/v1');p.add_argument('--model',default='glm-judge');a=p.parse_args()
    print(json.dumps(adjudicate(Path(__file__).parent,a.url,a.model),ensure_ascii=False,indent=2))
