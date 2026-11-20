"""Review all proposed onsets and their same-speaker event ancestry together."""
import copy,hashlib,json,time,urllib.request
from pathlib import Path
from event_extractor import finalize,speaker_gate
from stateful_bi import add_onset_features

RULES=Path(__file__).with_name('interaction_expanded_review.txt').read_text(encoding='utf8')
FIELDS={'candidate_id':{'type':'string'},'action':{'type':'string','enum':['KEEP','MERGE','DROP']},
 'label':{'type':'string','enum':['INTERRUPT','BACKCHANNEL']},'merge_into':{'type':'string'},'reason':{'type':'string'}}
SCHEMA={'type':'object','properties':{'decisions':{'type':'array','items':dict(type='object',properties=FIELDS,required=list(FIELDS),additionalProperties=False)}},'required':['decisions'],'additionalProperties':False}

def check_graph(candidates,judgment):
 byid={c['candidate_id']:c for c in candidates};ds=judgment['decisions']
 if len(ds)!=len(byid) or {d['candidate_id'] for d in ds}!=set(byid):raise ValueError('exact_candidate_coverage_required')
 mapped={d['candidate_id']:d for d in ds}
 for d in ds:
  c=byid[d['candidate_id']]
  if d['action']=='MERGE':
   parent=byid.get(d['merge_into']);pd=mapped.get(d['merge_into'])
   if not parent or parent['start']>=c['start']:raise ValueError('merge_parent_must_be_earlier:'+d['candidate_id'])
   if (parent['initiator'],parent['recipient'])!=(c['initiator'],c['recipient']):raise ValueError('cross_direction_merge:'+d['candidate_id'])
   if pd['action'] not in ('KEEP','MERGE') or pd['label']!=d['label']:raise ValueError('dropped_or_wrong_function_parent:'+d['candidate_id'])
  elif d['merge_into']:raise ValueError('non_merge_has_parent:'+d['candidate_id'])
 return judgment

def review(case,baseline,*,endpoint,model='qwen38-27b'):
 if baseline.get('status')!='ok':return baseline
 candidates=add_onset_features(copy.deepcopy(baseline['candidates']))
 aliases={c['candidate_id']:f'C{i}' for i,c in enumerate(candidates,1)};reverse={v:k for k,v in aliases.items()}
 rows=[]
 for c in candidates:
  row={k:v for k,v in c.items() if k not in ('word_ids','overlap_intervals','mix_feedback_supported','audio_neither_probability')}
  row['candidate_id']=aliases[c['candidate_id']]
  for key in ('previous_own_candidate','next_own_candidate'):
   if row.get(key):row[key]=aliases[row[key]]
  other=case['streams'][c['recipient']]
  row['other_word_active_at_exact_onset']=any(w['start']<c['start']<w['end'] for w in other)
  row['previous_other_words']=' '.join(w['text'] for w in other if c['start']-4<=w['end']<=c['start'])
  rows.append(row)
 provisional=copy.deepcopy(baseline['judgment']['decisions'])
 for d in provisional:
  d['candidate_id']=aliases[d['candidate_id']]
  if d['merge_into']:d['merge_into']=aliases[d['merge_into']]
 payload=dict(candidates=rows,provisional_decisions=provisional,onset_context=case['context'],
   complete_dialogue=[{k:v for k,v in b.items() if k not in ('word_timing','word_timing_columns')} for b in case.get('dialogue_context',case['context'])])
 attempts=[]
 for _ in range(3):
  messages=[dict(role='system',content=RULES),dict(role='user',content=json.dumps(payload,separators=(',',':')))]
  if attempts:
   error=attempts[-1]['error']
   for uid,alias in aliases.items():error=error.replace(uid,alias)
   messages.append(dict(role='user',content='Fix this graph validation failure; return all decisions: '+error))
  body=dict(model=model,messages=messages,temperature=0,max_tokens=4096,chat_template_kwargs={'enable_thinking':False},
    response_format={'type':'json_schema','json_schema':dict(name='expanded_event_review',strict=True,schema=SCHEMA)})
  start=time.monotonic();attempt=dict(stage='expanded_context_review',request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest());attempts.append(attempt)
  try:
   req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=360) as f:response=json.load(f)
   choice=response['choices'][0];attempt.update(seconds=time.monotonic()-start,usage=response.get('usage'),finish_reason=choice['finish_reason'])
   if choice['finish_reason']!='stop':raise ValueError('unfinished_generation')
   judgment=json.loads(choice['message']['content']);attempt['raw_judgment']=copy.deepcopy(judgment)
   for d in judgment['decisions']:
    d['candidate_id']=reverse[d['candidate_id']]
    if d['merge_into']:d['merge_into']=reverse[d['merge_into']]
   check_graph(candidates,judgment)
   result=finalize(case,candidates,judgment,speaker_gate(case,dict(mixed_audio_checked=True)))
   old={d['candidate_id']:d for d in baseline['judgment']['decisions']}
   reviews=[dict(candidate_id=d['candidate_id'],before=old[d['candidate_id']],after=d,
       changed=any(d[k]!=old[d['candidate_id']][k] for k in ('action','label','merge_into')),
       status='ok',review_scope='onset_and_event_chain',attempts=[]) for d in judgment['decisions']]
   result.update(status='ok',judgment=judgment,candidates=candidates,context_reviews=reviews,
      previous_context_reviews=baseline.get('context_reviews',[]),expanded_review_attempts=attempts,
      attempts=baseline.get('attempts',[])+attempts,conversation_id=case['conversation_id'],number=case['number'],target=case['target'],
      annotation_source='expanded_context_and_event_chain_review',context_policy=case.get('context_policy'),
      context_review_failed=False,audio_semantic_judge_used=True,audio_view='mix')
   return result
  except Exception as exc:attempt.update(error=str(exc),seconds=time.monotonic()-start)
 result=copy.deepcopy(baseline);result.update(expanded_review_attempts=attempts,context_review_failed=True,expanded_review_status='error_retained_original')
 result['attempts']=baseline.get('attempts',[])+attempts
 result['context_reviews']=[dict(candidate_id=c['candidate_id'],before=next(d for d in baseline['judgment']['decisions'] if d['candidate_id']==c['candidate_id']),
   after=next(d for d in baseline['judgment']['decisions'] if d['candidate_id']==c['candidate_id']),changed=False,status='error_retained_original',attempts=[]) for c in candidates]
 return result
