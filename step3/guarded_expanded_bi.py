"""Use graph review selectively; verify suspected positive events individually."""
import copy,hashlib,json,time,urllib.request
from pathlib import Path
from event_extractor import finalize,speaker_gate
RULES=Path(__file__).with_name('interaction_response_review.txt').read_text(encoding='utf8')
FIELDS={'finding':{'type':'string','enum':['UNCHANGED','NORMAL_RESPONSE','OWN_CONTINUATION']},
 'target_quote':{'type':'string'},'evidence_quote':{'type':'string'},'reason':{'type':'string'}}
SCHEMA=dict(type='object',properties=FIELDS,required=list(FIELDS),additionalProperties=False)

def canonicalize_non_event_links(judgment):
 ds={d['candidate_id']:d for d in judgment['decisions']}
 for d in ds.values():
  if d['action']!='MERGE':d['merge_into']=''
 while True:
  orphans=[d for d in ds.values() if d['action']=='MERGE' and ds.get(d['merge_into'],{}).get('action')=='DROP']
  if not orphans:break
  for d in orphans:d.update(action='DROP',merge_into='',reason='Continuation of dropped non-intervention')
 return judgment

def apply_graph_proposals(candidates,baseline,proposals):
 baseline=canonicalize_non_event_links(copy.deepcopy(baseline))
 byid={c['candidate_id']:c for c in candidates};old={d['candidate_id']:d for d in baseline['decisions']}
 if len(proposals['decisions'])!=len(byid) or {d['candidate_id'] for d in proposals['decisions']}!=set(byid):raise ValueError('proposal_coverage')
 proposed={d['candidate_id']:d for d in proposals['decisions']};selected=copy.deepcopy(old);changes=[]
 for c in sorted(candidates,key=lambda x:x['start']):
  uid=c['candidate_id'];d=old[uid];p=proposed[uid]
  # Keep receipt events and their established ancestry; onset removal is a separate audit.
  if d['action'] in ('KEEP','MERGE') and d['label']=='BACKCHANNEL':continue
  allow=False
  if p['action']=='MERGE' and p['label']=='INTERRUPT':
   parent=byid.get(p['merge_into']);pd=selected.get(p['merge_into'])
   allow=bool(parent and parent['start']<c['start'] and (parent['initiator'],parent['recipient'])==(c['initiator'],c['recipient'])
      and pd['action'] in ('KEEP','MERGE') and pd['label']=='INTERRUPT')
  elif p['action']=='KEEP' and p['label']=='INTERRUPT' and d['action'] in ('DROP','MERGE'):
   # Recover a fragmented launch supported by a proposed continuing clause.
   children=[q for q in proposals['decisions'] if q['action']=='MERGE' and q['merge_into']==uid and q['label']=='INTERRUPT'
      and byid[q['candidate_id']]['start']>c['start'] and (byid[q['candidate_id']]['initiator'],byid[q['candidate_id']]['recipient'])==(c['initiator'],c['recipient'])]
   allow=bool(children and c['other_active_run_at_onset'] and not c['text'].rstrip().endswith(('.', '?','!','。','？','！')))
  if allow:
   selected[uid]=copy.deepcopy(p)
   if any(p[k]!=d[k] for k in ('action','label','merge_into')):changes.append(uid)
 return dict(decisions=list(selected.values())),changes

def remove_root(judgment,uid,reason):
 ds={d['candidate_id']:d for d in judgment['decisions']};removed={uid}
 while True:
  more={i for i,d in ds.items() if d['action']=='MERGE' and d['merge_into'] in removed}-removed
  if not more:break
  removed.update(more)
 for i in removed:ds[i].update(action='DROP',label='BACKCHANNEL',merge_into='',reason=reason if i==uid else 'Continuation of ordinary non-intervention')
 return sorted(removed)

def removal_supported(case,candidate,decision,answer):
 # A receipt inside the recipient's active clause is not an ordinary free turn.
 if decision['label']=='BACKCHANNEL' and candidate.get('recipient_clause_at_onset'):
  return False,'active_recipient_clause_preserves_feedback'
 if answer['finding']=='NORMAL_RESPONSE':
  # Verify that the quoted response source was available by entry, allowing
  # only the established 350 ms final-word transition tolerance.
  prefix=' '.join(w['text'] for w in case['streams'][candidate['recipient']]
      if w['end']<=candidate['start']+.35)
  if answer['evidence_quote'] not in prefix:
   return False,'response_source_not_completed_at_entry'
 return True,None

def review(case,baseline,proposal_result,*,endpoint,model='qwen38-27b'):
 if baseline.get('status')!='ok':return baseline
 candidates=copy.deepcopy(proposal_result.get('candidates',baseline['candidates']))
 from stateful_bi import add_onset_features
 add_onset_features(candidates)
 reverse={f'C{i}':c['candidate_id'] for i,c in enumerate(candidates,1)}
 attempts=copy.deepcopy(proposal_result.get('expanded_review_attempts',[]));raw=next((a['raw_judgment'] for a in reversed(attempts) if a.get('raw_judgment')),None)
 proposal_error=None
 try:
  if not raw:raise ValueError('no_complete_graph_proposal')
  proposals=copy.deepcopy(raw)
  for d in proposals['decisions']:
   d['candidate_id']=reverse[d['candidate_id']]
   if d['merge_into']:d['merge_into']=reverse[d['merge_into']]
  judgment,graph_changes=apply_graph_proposals(candidates,baseline['judgment'],proposals)
 except Exception as e:
  judgment=copy.deepcopy(baseline['judgment']);graph_changes=[];proposal_error=str(e)
 decisions={d['candidate_id']:d for d in judgment['decisions']};audits=[]
 full=[{k:v for k,v in b.items() if k not in ('word_timing','word_timing_columns')} for b in case.get('dialogue_context',case['context'])]
 evidence='\n'.join(b['text'] for b in full)+'\n'+'\n'.join(b['text'] for b in case['context'])
 for c in sorted(candidates,key=lambda x:x['start']):
  d=decisions[c['candidate_id']]
  if d['action']!='KEEP':continue
  # All positive interruptions; feedback when a recent question could make it an answer.
  recent_question=any('?' in b['text'] and b['start']<c['start'] and b['end']>=c['start']-8 for b in case['context'])
  if d['label']!='INTERRUPT' and not recent_question:continue
  row=dict(candidate_id=c['candidate_id'],before=copy.deepcopy(d),attempts=[],changed=False)
  payload=dict(target={k:v for k,v in c.items() if k not in ('word_ids','overlap_intervals')},current_event=d,onset_context=case['context'],complete_dialogue=full)
  for _ in range(2):
   body=dict(model=model,temperature=0,max_tokens=500,chat_template_kwargs={'enable_thinking':False},messages=[dict(role='system',content=RULES),dict(role='user',content=json.dumps(payload,separators=(',',':')))],response_format={'type':'json_schema','json_schema':dict(name='positive_event_audit',strict=True,schema=SCHEMA)})
   start=time.monotonic();attempt=dict(stage='positive_event_audit',candidate_id=c['candidate_id'],request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest());row['attempts'].append(attempt)
   try:
    req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=360) as f:response=json.load(f)
    choice=response['choices'][0];attempt.update(seconds=time.monotonic()-start,usage=response.get('usage'),finish_reason=choice['finish_reason'])
    if choice['finish_reason']!='stop':raise ValueError('unfinished_generation')
    answer=json.loads(choice['message']['content']);attempt['answer']=answer
    if answer['finding']!='UNCHANGED':
     if not answer['target_quote'] or answer['target_quote'] not in c['text'] or not answer['evidence_quote'] or answer['evidence_quote'] not in evidence:raise ValueError('evidence_not_grounded')
     supported,blocked=removal_supported(case,c,d,answer)
     row['removal_blocked_reason']=blocked
     if supported:
      row['removed_candidates']=remove_root(judgment,c['candidate_id'],answer['reason']);row['changed']=True
    row['finding']=answer['finding'];break
   except Exception as e:attempt.update(error=str(e),seconds=time.monotonic()-start)
  row.update(after=copy.deepcopy(d),status='ok' if 'finding' in row else 'error_retained_original');audits.append(row)
 from expanded_review_bi import check_graph
 canonicalize_non_event_links(judgment);check_graph(candidates,judgment)
 result=finalize(case,candidates,judgment,speaker_gate(case,dict(mixed_audio_checked=True)))
 old={d['candidate_id']:d for d in baseline['judgment']['decisions']};reviewed={r['candidate_id']:r for r in audits}
 reviews=[]
 for d in judgment['decisions']:
  uid=d['candidate_id'];a=reviewed.get(uid,{})
  reviews.append(dict(candidate_id=uid,before=old[uid],after=d,changed=any(old[uid][k]!=d[k] for k in ('action','label','merge_into')),
      status=a.get('status','error_retained_original' if proposal_error else 'ok'),review_scope='graph_and_response_audit',attempts=a.get('attempts',[])))
 result.update(status='ok',candidates=candidates,judgment=judgment,context_reviews=reviews,previous_context_reviews=baseline.get('context_reviews',[]),
    graph_changes=graph_changes,graph_proposal_error=proposal_error,expanded_review_attempts=attempts,response_audits=audits,
    attempts=baseline.get('attempts',[])+attempts+[a for r in audits for a in r['attempts']],conversation_id=case['conversation_id'],number=case['number'],target=case['target'],
    annotation_source='guarded_graph_and_response_review',context_review_failed=bool(proposal_error or any(r['status']!='ok' for r in audits)),context_policy=case.get('context_policy'),audio_semantic_judge_used=True,audio_view='mix')
 return result
