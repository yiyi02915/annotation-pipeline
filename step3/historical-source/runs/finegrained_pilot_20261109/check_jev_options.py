"""Small diagnostic: change only label-option wording, not definitions or evidence."""
import collections,json,random,time
from benchmark_common import ROOT,read_inputs,context_text,LABELS
import torch
from duplexjev import Decider,Question
OPTIONS={
 'overlap':{'BACKCHANNEL':'listener feedback supporting the other speaker without taking over',
            'INTERRUPT':'trying to interrupt or take over before the other speaker finishes',
            'OTHER':'ordinary overlap or continuing an already held turn, neither of the above',
            'UNCERTAIN':'not enough evidence to decide'},
 'pause':{'HESITATION':'hesitating or planning while retaining the unfinished turn',
          'TURN_TRANSITION':'the prior turn ended or yielded and a new turn begins',
          'OTHER':'a breath or rhetorical pause, or suspension due to interruption',
          'UNCERTAIN':'not enough evidence to decide'}}
rows=read_inputs();chosen=[];rng=random.Random(20261110)
for task in LABELS:chosen+=rng.sample([e for e in rows if e['task']==task],100)
t=time.perf_counter();engine=Decider.from_pretrained(str(ROOT.parents[1]/'models/DuplexJev-4B-Para'),device='cuda:0',attn_implementation='sdpa')
torch.cuda.synchronize();load=time.perf_counter()-t
results=[]
for task in LABELS:
    choices=OPTIONS[task];rev={v:k for k,v in choices.items()}
    wording=("For the identified FOCUS entry into the OTHER speaker's speech, which observed behavior applies?" if task=='overlap'
              else 'For the specified pause between LEFT and RIGHT of FOCUS, which observed behavior applies?')
    qs=[Question('label',wording,list(choices.values()),audio='*'),
        Question('turn',"What is the speaker's turn state?",['finished, ready for the other speaker','not finished, still talking','just a backchannel, not taking the turn','hesitating or asking to wait'],audio='*'),
        Question('barge_in','Is the speaker trying to interrupt or take over the conversation?',['yes','no'],audio='*')]
    subset=[e for e in chosen if e['task']==task]
    for offset in range(0,len(subset),4):
        batch=subset[offset:offset+4]
        contexts={e['id']:context_text(e)+'\nThe audio is the isolated FOCUS channel. Other-speaker speech is provided in the context; do not attribute it to FOCUS.' for e in batch}
        answers=engine.decide_batch({e['id']:e['audio_clip'] for e in batch},qs,context=contexts,lang='en',mode='packed',max_items=4)
        for e in batch:
            answer=answers[e['id']]['label']
            results.append({'id':e['id'],'task':task,'label':rev[answer['answer']],'confidence':answer['confidence'],
                            'probs':{rev[k]:v for k,v in answer['probs'].items()}})
    print('OPTION_CHECK',task,len(subset),flush=True)
with (ROOT/'jev_option_check.jsonl').open('w') as f:
    for r in results:f.write(json.dumps(r)+'\n')
(ROOT/'jev_option_check_metadata.json').write_text(json.dumps({'count':200,'per_task':100,'seed':20261110,
    'changed':'primary question option wording only; same definitions, audio, context and auxiliary questions',
    'options':OPTIONS,'model_load_seconds':load,'accuracy_evaluated':False},indent=2))
print('OPTION_CHECK_COMPLETE',flush=True)
