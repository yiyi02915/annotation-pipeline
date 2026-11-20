import os,json,time,hashlib,argparse,urllib.request,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];B=D.parent/'refine205_20261109'
LABELS=['PAUSE','TURN_TRANSITION']
RULES='''Annotate the single marked GAP between two consecutive words of FOCUS. Do not edit, merge or split any transcript/audio. Mandatory binary operational taxonomy based on contextual meaning:
PAUSE: after the gap, FOCUS continues, completes, elaborates, explains or repairs the SAME ongoing communicative contribution. A full sentence before the gap can still belong to an ongoing contribution. Brief listener feedback from OTHER does not itself end that contribution. PAUSE is a semantic continuation label, NOT a claim of psychological hesitation.
TURN_TRANSITION: the preceding contribution has ended and the following speech starts a distinct contribution or responds to OTHER's intervening substantive contribution. In this dataset this includes a semantic change of contribution without requiring a change of speaker ID. Overall dialogue may remain topically coherent across this boundary.
Decide semantic dependency, not simply same topic, conjunctions, punctuation, gap length, grammatical completeness, or whether another voice is audible. Determine what FOCUS's first speech after RIGHT primarily connects to: their preceding ongoing expression, or a newly initiated contribution / OTHER's new content. For short acknowledgments, examine what each acknowledgment responds to; do not assume all acknowledgments are one contribution or all are new turns. Use longer context to interpret the exact LEFT-to-RIGHT gap, not a different pause or the clip endpoint.
Exactly one of PAUSE or TURN_TRANSITION is required. Never output OTHER, UNCERTAIN, an abstention or multiple labels. If evidence is imperfect, choose the better-supported binary label; technical quality is recorded separately. No old model labels are provided.'''
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def events():return read(B/'events.jsonl')
def payload(e,view='a'):
 src=e['payload'];p={k:v for k,v in src.items() if k not in ['target','independent_asr']};p['target']={k:v for k,v in src['target'].items() if k!='boundary_stable'};p['independent_asr']=src['independent_asr']
 if view=='b':
  p={'target':p['target'],'focus':p['focus'],'other':p['other'],'focus_before_target':p['focus_before_target'],'focus_after_target':p['focus_after_target'],'other_near_target':p['other_near_target'],'context':p['context'],'independent_asr':p['independent_asr'],'audio_absolute_interval':p['audio_absolute_interval']}
 return p
SCHEMA={'type':'object','properties':{'label':{'type':'string','enum':LABELS},'before_meaning':{'type':'string'},'after_meaning':{'type':'string'},'semantic_link':{'type':'string'},'other_speaker_effect':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string'}}},'required':['label','before_meaning','after_meaning','semantic_link','other_speaker_effect','evidence_ids'],'additionalProperties':False}
def request(e,view,port):
 t=time.time();attempts=[];extra={'a':'Read the complete interaction, then identify the meaning of FOCUS just before and just after the marked gap. Decide whether the same contribution continues. Keep each explanation concise and cite actual context segment_id values.', 'b':'First examine what RIGHT is responding to. Then trace backward to LEFT and inspect OTHER near the gap. Use semantic dependency to choose exactly one binary label. Explain why the chosen relationship fits the marked gap; cite actual segment_id values.', 'arbiter':'Reassess this raw event from scratch. Determine whether the post-gap contribution semantically continues the preceding one or initiates/responds with a distinct contribution. Do not mistake topical continuity for continuation of one contribution, or grammatical sentence completion for contribution closure. Choose exactly one binary label with specific contextual evidence.'}[view]
 for attempt in range(2):
  thinking=(view in ['a','arbiter']) and attempt==0
  body={'model':'qwen38-27b','messages':[{'role':'system','content':RULES+'\n'+extra+' Return JSON; each explanation should be at most 70 words.'},{'role':'user','content':json.dumps(payload(e,view))}],'temperature':0,'max_tokens':(4096 if view=='arbiter' else 3072) if thinking else 1536,'chat_template_kwargs':{'enable_thinking':thinking},'response_format':{'type':'json_schema','json_schema':{'name':'binary_gap_semantics','strict':True,'schema':SCHEMA}}}
  try:
   req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=1200) as response:raw=json.load(response)
   c=raw['choices'][0];attempts.append({'thinking':thinking,'finish_reason':c['finish_reason'],'usage':raw.get('usage'),'content':c['message'].get('content'),'reasoning':c['message'].get('reasoning_content',c['message'].get('reasoning'))})
   v=json.loads(c['message']['content']);assert v['label'] in LABELS and c['finish_reason']=='stop'
   valid={s['segment_id'] for s in e['payload']['context']};refs=v['evidence_ids'];v['references_valid']=bool(refs) and all(x in valid for x in refs)
   return {'id':e['id'],'view':view,'status':'ok','judgment':v,'attempts':attempts,'seconds':time.time()-t,'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
  except Exception as ex:attempts.append({'error':str(ex)})
 return {'id':e['id'],'view':view,'status':'error','attempts':attempts,'seconds':time.time()-t}
def qwen(shard):
 port=18550+shard
 while True:
  try:urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3).close();break
  except Exception:time.sleep(5)
 out=D/f'qwen_{shard}.jsonl';done={(r['id'],r['view']) for r in read(out) if r['status']=='ok'};jobs=[(e,v) for i,e in enumerate(events()) if i%3==shard for v in ['a','b'] if (e['id'],v) not in done]
 with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  pending=[pool.submit(request,e,v,port) for e,v in jobs]
  for future in concurrent.futures.as_completed(pending):z=future.result();f.write(json.dumps(z)+'\n');f.flush();print('QWEN',z['view'],z['status'],z['id'],flush=True)
 print('QWEN_COMPLETE',flush=True)
def jev():
 from duplexjev import Decider,Question
 engine=Decider.from_pretrained(str(R/'models/DuplexJev-32B-Para'),device='cuda:0',attn_implementation='sdpa');print('MODEL_READY',flush=True)
 class StableQuestion(Question):
  def permutation(self,seed=0):return Question('label','',LABELS,audio=self.audio).permutation(seed)
 qs=[StableQuestion('label','At the exact marked FOCUS gap, does the meaning after RIGHT continue the same contribution (PAUSE) or begin/respond with a distinct contribution (TURN_TRANSITION)? Apply the operational semantic definitions, not an acoustic endpoint test.',LABELS,audio='*'),StableQuestion('check','Trace the post-gap expression back through the context: is it completing/elaborating/repairing the pre-gap contribution, or is the preceding contribution closed and this a distinct one? Choose PAUSE for the first and TURN_TRANSITION for the second.',LABELS,audio='*')]
 out=D/'jev.jsonl';done={x['id'] for x in read(out) if x['status']=='ok'}
 with out.open('a') as f:
  for e in events():
   if e['id'] in done:continue
   t=time.time()
   try:
    result=engine.decide(e['mixed_audio'],qs,context=RULES+'\nAudio contains BOTH speakers in original time order. Semantic context is primary; audio supports target localization and the function of speech.\n'+json.dumps(payload(e)),lang='en',mode='packed',n_perm=4,seed=0)
    z={'id':e['id'],'status':'ok','answers':result,'seconds':time.time()-t}
   except Exception as ex:z={'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
   f.write(json.dumps(z)+'\n');f.flush();print('JEV',len(done)+1,z['status'],flush=True);done.add(e['id'])
 print('JEV_COMPLETE',flush=True)
def arbitrate():
 while not all((D/f'qwen_{k}.log').exists() and 'QWEN_COMPLETE' in (D/f'qwen_{k}.log').read_text() for k in range(3)) or not (D/'jev.log').exists() or 'JEV_COMPLETE' not in (D/'jev.log').read_text():time.sleep(5)
 q={(x['id'],x['view']):x for k in range(3) for x in read(D/f'qwen_{k}.jsonl')};j={x['id']:x for x in read(D/'jev.jsonl')};selected=[]
 for i,e in enumerate(events()):
  a=q.get((e['id'],'a'),{});b=q.get((e['id'],'b'),{});z=j.get(e['id'],{});ja=z.get('answers',{})
  labels=[a.get('judgment',{}).get('label'),b.get('judgment',{}).get('label'),ja.get('label',{}).get('answer'),ja.get('check',{}).get('answer')]
  if len(set(labels))!=1 or None in labels or not a.get('judgment',{}).get('references_valid') or not b.get('judgment',{}).get('references_valid'):selected.append((e,18550+i%3))
 out=D/'arbitration.jsonl';done={x['id'] for x in read(out) if x['status']=='ok'}
 (D/'arbitration_selection.json').write_text(json.dumps({'ids':[e['id'] for e,p in selected],'rule':'Any disagreement across two text views and two audio question forms, or invalid citation; fresh blind semantic judgment. Same text model, not independent gold.'},indent=2))
 with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
  pending=[pool.submit(request,e,'arbiter',port) for e,port in selected if e['id'] not in done]
  for future in concurrent.futures.as_completed(pending):z=future.result();f.write(json.dumps(z)+'\n');f.flush();print('ARBITRATE',z['status'],z['id'],flush=True)
 print('ARBITRATION_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
 if a.mode=='qwen':qwen(a.shard)
 elif a.mode=='jev':jev()
 elif a.mode=='arbitrate':arbitrate()
