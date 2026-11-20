import time,json,torch,argparse
from evidence import *
from duplexjev import Decider,Question
pa=argparse.ArgumentParser();pa.add_argument('--empty-fix',action='store_true');args=pa.parse_args()
selected={e['id'] for e in read(D/'inputs.jsonl') if not args.empty_fix or any(not norm(tr['text']) for tr in e['tracks'].values())}
engine=Decider.from_pretrained(str(D.parents[1]/'models/DuplexJev-32B-Para'),device='cuda:0',attn_implementation='sdpa');print('MODEL_READY',flush=True)
class StableQuestion(Question):
 def permutation(self,seed=0):return Question('label','',LABELS,audio=self.audio).permutation(seed)
qs=[StableQuestion('label','Which label describes the specific FOCUS pause between LEFT and RIGHT in the timeline?',LABELS,audio='*'),StableQuestion('check','Using both speakers before and after the specified gap, classify whether FOCUS is hesitating within a held contribution, moving to a new turn, another behavior, or whether evidence is insufficient.',LABELS,audio='*')]
out=D/('jev_rejudge_emptyfix.jsonl' if args.empty_fix else 'jev_rejudge.jsonl');started=time.time()
while True:
 done={e['id'] for e in read(out)};batch=[x for x in ready() if x[0]['id'] in selected and x[0]['id'] not in done]
 with out.open('a') as f:
  for e,p in batch:
   t=time.time()
   try:
    context=DEFINITIONS+'\n'+EXTRA+'\nThe audio contains BOTH speakers mixed at equal gain in original time order. Speaker identities and utterance timing are in the transcript. Do not confuse OTHER with FOCUS. Audio begins at absolute time '+str(e['expanded_interval'][0])+'.\n'+json.dumps(p)
    v=engine.decide(e['mixed_audio'],qs,context=context,lang='en',mode='packed',n_perm=4,seed=0)
    r={'id':e['id'],'status':'ok','answers':v,'seconds':time.time()-t}
   except Exception as ex:r={'id':e['id'],'status':'error','error':str(ex),'seconds':time.time()-t}
   f.write(json.dumps(r)+'\n');f.flush();done.add(e['id']);print('JEV',len(done),r['status'],flush=True)
 if selected<=done:break
 if time.time()-started>14400:raise TimeoutError('Incomplete inputs')
 time.sleep(10)
print('JEV_COMPLETE',flush=True)
