import os,sys,json,time,hashlib,argparse,urllib.request,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1];B=D.parent/'rejudge291_20261109'
def read(p):
 if not p.exists():return []
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
LABELS=['HESITATION','TURN_TRANSITION','OTHER','UNCERTAIN']
ROLES=['feedback_only','substantive_turn','interrupting','none','unclear']
RULES='''Classify ONLY the marked pause of FOCUS, not the end of the audio or a different speaker's pause. LEFT and RIGHT identify the exact two consecutive words; their clip-relative times locate the event. Use the longer context only to interpret that event. HESITATION: FOCUS was contributing, pauses for planning/search/self-repair within an unfinished contribution, then resumes that contribution; brief feedback does not transfer the floor. TURN_TRANSITION: FOCUS's previous contribution ended/yielded, OTHER has a substantive turn, and FOCUS returns with a new turn. OTHER: ordinary breathing/rhetorical pause without hesitation, suspension caused by interruption, or a listener's gap between brief backchannels while OTHER retains the floor. Consecutive acknowledgments by a listener are NOT evidence of hesitation or two floor transfers. UNCERTAIN: evidence insufficient or ambiguous. Grammatical completion, punctuation, duration, and the mere presence of OTHER's speech do not determine the label. Other's function must be evaluated around the TARGET interval; speech elsewhere is not proof of a turn at this gap. Do not infer intention unsupported by transcript/audio. Distinguish the focus's role before the pause, the other speaker's function during it, and the focus's contribution after it.'''
def prepare():
 import numpy as np,soundfile as sf
 rows={e['id']:e for e in read(B/'inputs.jsonl')};rep={r['id']:r for r in read(D/'alignment_repairs.jsonl')};out=[];(D/'clips').mkdir(exist_ok=True)
 for old in read(B/'adjudicated.jsonl'):
  issue=old.get('primary_remaining_issue','')
  if issue not in ['对齐或停顿边界不稳定','重判模型仍分歧']:continue
  e=rows[old['id']];fix=rep.get(e['id']);boundary=(fix or {}).get('accepted_boundary');anchor=old['alignment']['anchor']
  stable=bool(boundary) if fix else True
  l=boundary['left_end'] if boundary else (anchor.get('left') or {}).get('end',e['anchor']);r=boundary['right_start'] if boundary else (anchor.get('right') or {}).get('start',e['event_end'])
  a=max(e['expanded_interval'][0],min(l,e['left_word']['start'])-4);b=min(e['expanded_interval'][1],max(r,e['right_word']['end'])+4)
  name=hashlib.sha256(e['id'].encode()).hexdigest()[:20];tracks={};arrays=[]
  for sp,path in e['audio_paths'].items():
   sr=sf.info(path).samplerate;x,_=sf.read(path,start=int(a*sr),stop=int(b*sr),dtype='float32');x=x.mean(axis=1) if x.ndim>1 else x;dest=D/'clips'/f'{name}_{sp}.wav';sf.write(dest,x,sr);tracks[sp]=str(dest);arrays.append(x)
  n=min(map(len,arrays));mix=D/'clips'/f'{name}_both.wav';sf.write(mix,sum(x[:n] for x in arrays)/len(arrays),sr)
  p={'id':e['id'],'focus':e['focus'],'other':e['other'],'target':{'left_word':e['left_word']['text'],'right_word':e['right_word']['text'],'left_word_id':e['left_word']['id'],'right_word_id':e['right_word']['id'],'absolute_gap':[round(l,3),round(r,3)],'clip_relative_gap':[round(l-a,3),round(r-a,3)],'gap_seconds':round(r-l,3),'boundary_stable':stable},'audio_absolute_interval':[a,b],'context':old['evidence']['context'],'independent_asr':old['evidence']['independent_asr']}
  ws=e['tracks'][e['focus']]['words'];idx={w['id']:i for i,w in enumerate(ws)};li=idx[e['left_word']['id']];ri=idx[e['right_word']['id']]
  p['focus_before_target']=' '.join(w['text'] for w in ws[max(0,li-24):li+1]);p['focus_after_target']=' '.join(w['text'] for w in ws[ri:ri+25]);p['other_near_target']=[s for s in p['context'] if s['speaker']==e['other'] and s['end']>=l-.3 and s['start']<=r+.3]
  out.append({'id':e['id'],'prior_issue':issue,'alignment_repair':fix,'boundary_stable':stable,'gap':round(r-l,3),'payload':p,'mixed_audio':str(mix),'audio_tracks':tracks,'old_reasons':old['reasons']})
 with (D/'events.jsonl').open('w') as f:
  for row in out:f.write(json.dumps(row)+'\n')
 print('PREPARED',len(out),flush=True)
def jev():
 from duplexjev import Decider,Question
 engine=Decider.from_pretrained(str(R/'models/DuplexJev-32B-Para'),device='cuda:0',attn_implementation='sdpa');print('MODEL_READY',flush=True)
 class StableQuestion(Question):
  def permutation(self,seed=0):return Question('label','',LABELS,audio=self.audio).permutation(seed)
 qs=[StableQuestion('label','Classify the marked FOCUS gap between LEFT and RIGHT using the supplied taxonomy. Ignore the final endpoint of this clip.',LABELS,audio='*'),StableQuestion('check','At the marked time interval, is FOCUS searching within their contribution, returning after an actual floor exchange, showing another pause behavior, or is it unclear?',LABELS,audio='*'),Question('other_role','What is OTHER doing during the marked FOCUS gap (including an utterance crossing its edges)? Brief acknowledgment is feedback; a meaningful answer, question, elaboration or account is substantive; competing before completion to seize the floor is interrupting. No relevant OTHER speech is none.',ROLES,audio='*')]
 while not (D/'events.jsonl').exists():time.sleep(5)
 out=D/'jev.jsonl';done={x['id'] for x in read(out)}
 with out.open('a') as f:
  for e in read(D/'events.jsonl'):
   if e['id'] in done:continue
   t=time.time()
   try:
    p=e['payload'];context=RULES+'\nAudio mixes BOTH speakers in their original time relationship. Transcript identifies their roles.\n'+json.dumps(p)
    v=engine.decide(e['mixed_audio'],qs,context=context,lang='en',mode='packed',n_perm=4,seed=0)
    role=engine.decide(e['audio_tracks'][p['other']],[qs[2]],context=RULES+'\nThis audio is ONLY OTHER, not FOCUS. The timeline below still describes both speakers. Judge OTHER only inside the marked interval, using its surrounding speech for meaning.\n'+json.dumps(p),lang='en',mode='packed',n_perm=4,seed=0)
    z={'id':e['id'],'status':'ok','answers':v,'isolated_other':role,'seconds':time.time()-t}
   except Exception as ex:z={'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
   f.write(json.dumps(z)+'\n');f.flush();done.add(e['id']);print('JEV',len(done),z['status'],flush=True)
 print('JEV_COMPLETE',flush=True)
def qwen(shard):
 enums={'label':LABELS,'focus_before_role':['contributor','listener_feedback','unclear'],'left_state':['unfinished','completed_or_yielded','unclear'],'other_action':ROLES,'focus_continuation':['same_contribution','new_turn','listener_feedback','unclear'],'pause_function':['planning_or_repair','turn_exchange','listening_between_feedback','breath_or_rhetorical','interruption_suspension','unclear']}
 props={k:{'type':'string','enum':v} for k,v in enums.items()};props.update({'evidence_ids':{'type':'array','items':{'type':'string'}},'target_explanation':{'type':'string'},'other_function_explanation':{'type':'string'}});schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
 port=18452+shard
 def call(e):
  t=time.time();body={'model':'qwen38-27b','messages':[{'role':'system','content':RULES+'\nAnalyze the exact marked gap; return consistent structured facts and label. Cite context segment_id evidence for FOCUS before and after and OTHER if present. Explain how LEFT and RIGHT belong to the target pause, and why OTHER has the reported function. Do not turn incomplete evidence into certainty.'},{'role':'user','content':json.dumps(e['payload'])}],'temperature':0,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':True},'reasoning_effort':'medium','response_format':{'type':'json_schema','json_schema':{'name':'target_gap','strict':True,'schema':schema}}}
  try:
   req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=1200) as response:a=json.load(response)
   c=a['choices'][0];v=json.loads(c['message']['content']);return {'id':e['id'],'status':'ok','judgment':v,'finish_reason':c['finish_reason'],'seconds':time.time()-t,'usage':a.get('usage'),'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
  except Exception as ex:return {'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
 while not (D/'events.jsonl').exists():time.sleep(5)
 while True:
  try:urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3).close();break
  except Exception:time.sleep(5)
 out=D/f'qwen_{shard}.jsonl';done={e['id'] for e in read(out) if e['status']=='ok'};events=[e for i,e in enumerate(read(D/'events.jsonl')) if i%2==shard and e['id'] not in done]
 with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  for z in pool.map(call,events):f.write(json.dumps(z)+'\n');f.flush();print('QWEN',z['status'],z['id'],flush=True)
 print('QWEN_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--shard',type=int,default=0);args=p.parse_args()
 if args.mode=='prepare':prepare()
 elif args.mode=='jev':jev()
 elif args.mode=='qwen':qwen(args.shard)
