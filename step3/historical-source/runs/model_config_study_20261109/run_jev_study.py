import argparse,json,time,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'finegrained_pilot_20261109';sys.path.insert(0,str(B))
CONFIG=json.loads((D/'annotation_config.candidate.json').read_text()) if (D/'annotation_config.candidate.json').exists() else {'duplexjev':{'mode':'packed','seed':0}}
from benchmark_common import LABELS,context_text
OPTIONS={
 'overlap':{'BACKCHANNEL':'listener feedback supporting the other speaker without taking over','INTERRUPT':'trying to interrupt or take over before the other speaker finishes','OTHER':'ordinary overlap or continuing an already held turn, neither of the above','UNCERTAIN':'not enough evidence to decide'},
 'pause':{'HESITATION':'hesitating or planning while retaining the unfinished turn','TURN_TRANSITION':'the prior turn ended or yielded and a new turn begins','OTHER':'a breath or rhetorical pause, or suspension due to interruption','UNCERTAIN':'not enough evidence to decide'}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--size',default='4B');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--perms',default='1,4');p.add_argument('--wordings',default='enum,alignednatural');p.add_argument('--all-events',action='store_true');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
 import torch
 from duplexjev import Decider,Question
 rows=[json.loads(x) for x in (D/'study_inputs.jsonl').read_text().splitlines()]
 if a.all_events:
  rows=[dict(json.loads(x),study_split='full_population') for x in (B/'benchmark_inputs.jsonl').read_text().splitlines()]
  assert 0<=a.shard_index<a.shards
  rows=rows[a.shard_index::a.shards]
 suffix=f'_full_shard{a.shard_index}' if a.all_events else ''
 start=time.perf_counter();engine=Decider.from_pretrained(str(D.parents[1]/'models'/f'DuplexJev-{a.size}-Para'),device='cuda:0',attn_implementation='sdpa');torch.cuda.synchronize();load=time.perf_counter()-start
 print('MODEL_LOADED',a.size,load,flush=True)
 runtimes=[]
 for perm in map(int,a.perms.split(',')):
  for wording in a.wordings.split(','):
   output=D/f'jev{a.size}_{wording}_p{perm}{suffix}.jsonl';done=set()
   if output.exists():done={json.loads(s)['id'] for s in output.read_text().splitlines()}
   with output.open('a') as f:
    for task in LABELS:
     choices={k:k for k in LABELS[task]} if wording=='enum' else OPTIONS[task];rev={v:k for k,v in choices.items()}
     question="For the identified FOCUS entry into the OTHER speaker's speech, which observed behavior applies?" if task=='overlap' else 'For the specified pause between LEFT and RIGHT of FOCUS, which observed behavior applies?'
     class AlignedQuestion(Question):
      def permutation(self,seed=0):
       return Question(self.id,self.text,LABELS[task],audio=self.audio).permutation(seed)
     primary=AlignedQuestion if wording=='alignednatural' else Question
     qs=[primary('label',question,list(choices.values()),audio='*'),Question('turn',"What is the speaker's turn state?",['finished, ready for the other speaker','not finished, still talking','just a backchannel, not taking the turn','hesitating or asking to wait'],audio='*'),Question('barge_in','Is the speaker trying to interrupt or take over the conversation?',['yes','no'],audio='*')]
     subset=[e for e in rows if e['task']==task and e['id'] not in done];ts=time.time();t=time.perf_counter()
     for offset in range(0,len(subset),a.batch_size):
      batch=subset[offset:offset+a.batch_size];contexts={e['id']:context_text(e)+'\nThe audio is the isolated FOCUS channel. Other-speaker speech is provided in the context; do not attribute it to FOCUS.' for e in batch}
      torch.cuda.synchronize();bt=time.perf_counter()
      answers=engine.decide_batch({e['id']:e['audio_clip'] for e in batch},qs,context=contexts,lang='en',mode=CONFIG['duplexjev']['mode'],n_perm=perm,seed=CONFIG['duplexjev']['seed'],max_items=a.batch_size)
      torch.cuda.synchronize();elapsed=time.perf_counter()-bt
      for e in batch:
       answer=answers[e['id']]['label'];f.write(json.dumps(dict(id=e['id'],task=task,split=e['study_split'],label=rev[answer['answer']],confidence=answer['confidence'],probs={rev[k]:v for k,v in answer['probs'].items()},batch_wall_seconds=elapsed,batch_size=len(batch),size=a.size,n_perm=perm,wording=wording))+'\n')
      f.flush()
     elapsed=time.perf_counter()-t;runtimes.append(dict(task=task,wording=wording,n_perm=perm,n=len(subset),seconds=elapsed,started_at=ts,finished_at=time.time()));print(runtimes[-1],flush=True)
   runtime_name=f'jev{a.size}_runtime.json' if a.wordings in ('enum,alignednatural','enum,natural,alignednatural') else f'jev{a.size}_{a.wordings.replace(",","_")}_runtime.json'
   if a.all_events:runtime_name=f'jev{a.size}{suffix}_runtime.json'
   (D/runtime_name).write_text(json.dumps(dict(load_seconds=load,runs=runtimes,includes_first_inference=True),indent=2))
 print('STUDY_COMPLETE',a.size,flush=True)
if __name__=='__main__':main()
