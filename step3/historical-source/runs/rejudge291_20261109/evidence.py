import json,re,difflib,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'finegrained_pilot_20261109'))
from benchmark_common import DEFINITIONS
LABELS=['HESITATION','TURN_TRANSITION','OTHER','UNCERTAIN']
def read(p):
 if not p.exists():return []
 out=[]
 for s in p.read_text().splitlines():
  try:out.append(json.loads(s))
  except json.JSONDecodeError:pass
 return out
def norm(s):return re.findall(r'[a-z0-9]+',s.lower())
def aligned_map(old,new):
 # Map normalized lexical tokens, allowing contractions/punctuation to split differently.
 aa=[(t,i) for i,w in enumerate(old) for t in norm(w['text'])];bb=[(t,i) for i,w in enumerate(new) for t in norm(w['text'])]
 pairs={};hit=0
 for a,b,n in difflib.SequenceMatcher(None,[x[0] for x in aa],[x[0] for x in bb],autojunk=False).get_matching_blocks():
  for k in range(n):pairs.setdefault(aa[a+k][1],[]).append(bb[b+k][1]);hit+=1
 mapped={i:{'start':min(new[j]['start'] for j in js),'end':max(new[j]['end'] for j in js)} for i,js in pairs.items() if len(js)==len(norm(old[i]['text']))}
 return mapped,hit/len(aa) if aa else 1.0
def prepare(e,asr,fa):
 mapped={};diag={};segments=[];seq=0
 for sp,tr in e['tracks'].items():
  fw=fa.get('tracks',{}).get(sp,{});aw=asr.get('tracks',{}).get(sp,{})
  mp,cov=aligned_map(tr['words'],fw.get('words',[]));mapped[sp]=mp
  shifts=[max(abs(w['start']-mp[i]['start']),abs(w['end']-mp[i]['end'])) for i,w in enumerate(tr['words']) if i in mp]
  diag[sp]={'mapped_fraction':cov,'max_boundary_shift':max(shifts,default=None),'asr_similarity':aw.get('transcript_token_similarity'),'error':fw.get('error') or aw.get('error')}
  groups=[]
  for i,w in enumerate(tr['words']):
   w=dict(w,**mp.get(i,{}))
   if not groups or w['start']-groups[-1][-1]['end']>.5:groups.append([])
   groups[-1].append(w)
  for group in groups:
   seq+=1;segments.append({'segment_id':f'S{seq}','speaker':sp,'start':round(group[0]['start'],3),'end':round(group[-1]['end'],3),'text':' '.join(w['text'] for w in group)})
 focus=e['focus'];ws=e['tracks'][focus]['words'];mp=mapped[focus]
 ids={w['id']:i for i,w in enumerate(ws)};li=ids.get(e['left_word']['id']);ri=ids.get(e['right_word']['id']);left=mp.get(li);right=mp.get(ri)
 flags=[]
 if not left or not right:flags.append('anchor_unmapped')
 asr_words=[{'text':x,'start':k,'end':k+1} for k,x in enumerate(norm(asr.get('tracks',{}).get(focus,{}).get('asr_text','')))];asr_mp,_=aligned_map(ws,asr_words)
 if li not in asr_mp or ri not in asr_mp:flags.append('anchor_text_not_confirmed_by_asr')
 newgap=round(right['start']-left['end'],3) if left and right else None
 if newgap is not None and newgap<=.5:flags.append('gap_no_longer_eligible')
 for sp,tr in e['tracks'].items():
  if not tr['finite']:flags.append('invalid_audio')
  dd=diag[sp]
  if dd['error']:flags.append('audio_evidence_error')
  if dd['mapped_fraction']<.8:flags.append('poor_fa_lexical_mapping')
  if dd['asr_similarity'] is None or (dd['asr_similarity']<.6 and len(norm(tr['text']))>=5):flags.append('asr_transcript_disagreement')
 shift=max(abs(left['end']-e['anchor']),abs(right['start']-e['event_end'])) if left and right else None
 if left and right and (left['end']<=left['start'] or right['end']<=right['start']):flags.append('degenerate_anchor')
 diag['anchor']={'old_gap':e['gap_seconds'],'new_gap':newgap,'max_shift':shift,'status':'unmapped' if shift is None else 'shift_gt_0.2s' if shift>.2 else 'within_0.2s','left':left,'right':right}
 payload={'id':e['id'],'task':'pause','focus':focus,'other':e['other'],'original_event':[e['anchor'],e['event_end']],'reestimated_event':[left['end'],right['start']] if left and right else None,'left_word':e['left_word']['text'],'right_word':e['right_word']['text'],'alignment_diagnostics':diag,'quality_flags':sorted(set(flags)),'context':sorted(segments,key=lambda s:(s['start'],s['speaker'])),'independent_asr':{sp:v.get('asr_text','') for sp,v in asr.get('tracks',{}).items()},'audio_interval':e['expanded_interval']}
 return payload
def ready():
 asr={e['id']:e for k in range(2) for e in read(D/f'asr_{k}.jsonl')};fa={e['id']:e for k in range(2) for e in read(D/f'fa_{k}.jsonl')}
 return [(e,prepare(e,asr[e['id']],fa[e['id']])) for e in read(D/'inputs.jsonl') if e['id'] in asr and e['id'] in fa]
EXTRA='''This is a fresh adjudication. No earlier model labels are supplied. Determine the specified FOCUS pause, not every pause in the longer clip. OTHER includes a turn suspension caused by an interruption. A brief acknowledgment from OTHER does not establish a transition. Inspect the focus's continuation and the other speaker's substantive contribution. Forced alignment is an estimate, not gold; ASR disagreement can reflect either recognizer errors or transcript errors. Do not treat silence, question marks, grammatical completion or acoustic endpoint alone as proof. If the target words cannot be located reliably, or the reestimated gap no longer exceeds 0.5 seconds, choose UNCERTAIN rather than inventing the requested event.'''
