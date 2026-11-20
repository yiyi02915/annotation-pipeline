"""Local decision first; full-dialogue review applies narrowly grounded corrections."""
import copy, hashlib, json, re, time, urllib.request
from pathlib import Path
from event_extractor import finalize, speaker_gate
from stateful_bi import add_onset_features

RULES=Path(__file__).with_name('interaction_context_review.txt').read_text(encoding='utf8')
FIELDS={'finding':{'type':'string','enum':['UNCHANGED','NORMAL_ANSWER','NEW_FEEDBACK']},
        'target_quote':{'type':'string'},'recipient_quote':{'type':'string'},'reason':{'type':'string'}}
SCHEMA=dict(type='object',properties=FIELDS,required=list(FIELDS),additionalProperties=False)
# This conservative recovery vocabulary does not filter original model events.
# It prevents fragments ("I", "and", "do") being promoted to feedback by a reviewer.
RECEIPTS={'yes','yeah','yep','yup','mhm','mmhm','mm hmm','uh huh','okay','ok','all right',
          'right','exactly','sure','absolutely','indeed','wow','oh','hmm','i see','i know',
          'got it','makes sense','嗯','嗯嗯','哦','啊','对','是的','好的','明白','原来如此'}

def receipt_text(text):
    return re.sub(r'[^\w\s]',' ',text.lower()).strip()

def recipient_key(c):
    r=c.get('recipient_clause_at_onset')
    return (c['initiator'],c['recipient'],r['start'],r['end'],r['text']) if r else None

def feedback_parent(c,active):
    previous=active.get(recipient_key(c))
    return previous[0] if previous and 0<=c['start']-previous[1]<=.5 else ''

def eligible(c,d):
    if not c.get('other_active_run_at_onset') or c.get('overlap_seconds',0)<=0:return []
    # Preserve the existing terminal-answer guard when adding a recovery stage.
    if c.get('terminal_question_answer'):return []
    if d['action']=='KEEP' and d['label']=='INTERRUPT' and c.get('recipient_is_question') and 0<=c['recipient_tail_seconds']<=.35:
        return ['NORMAL_ANSWER']
    if d['action']=='DROP' and ' '.join(receipt_text(c['text']).split()) in RECEIPTS and c.get('recipient_clause_at_onset'):
        return ['NEW_FEEDBACK']
    return []

def apply_review(c,d,answer,allowed):
    if answer['finding']=='UNCHANGED':return False
    if answer['finding'] not in allowed:raise ValueError('correction_not_allowed')
    recipient=c.get('recipient_clause_at_onset') or {}
    for key,text in [('target_quote',c['text']),('recipient_quote',recipient.get('text',''))]:
        if not answer[key].strip() or answer[key] not in text:raise ValueError('evidence_quote_not_grounded:'+key)
    if answer['finding']=='NORMAL_ANSWER':d.update(action='DROP',merge_into='',reason=answer['reason'])
    else:d.update(action='KEEP',label='BACKCHANNEL',merge_into='',reason=answer['reason'])
    return True

def review(case,baseline,*,endpoint,model='qwen38-27b'):
    if baseline.get('status')!='ok':return baseline
    candidates=add_onset_features(copy.deepcopy(baseline['candidates']))
    judgment=copy.deepcopy(baseline['judgment']);decisions={d['candidate_id']:d for d in judgment['decisions']}
    reviews=[]
    active_receipts={}
    for c in sorted(candidates,key=lambda c:(c['start'],c['candidate_id'])):
        d=decisions[c['candidate_id']];allowed=eligible(c,d)
        key=recipient_key(c)
        if ' '.join(receipt_text(c['text']).split()) not in RECEIPTS:
            active_receipts={k:v for k,v in active_receipts.items() if k[:2]!=(c['initiator'],c['recipient'])}
        if d['action'] in ('KEEP','MERGE') and d['label']=='BACKCHANNEL' and key:
            active_receipts[key]=(d.get('merge_into') or c['candidate_id'],c['end'])
        # A parent with retained children cannot be deleted without rejudging the graph.
        if any(p.get('merge_into')==c['candidate_id'] for p in judgment['decisions']):continue
        if not allowed:continue
        row=dict(candidate_id=c['candidate_id'],allowed_corrections=allowed,before=copy.deepcopy(d),attempts=[],changed=False)
        payload=dict(target={k:v for k,v in c.items() if k not in ('word_ids','overlap_intervals')},
            allowed_corrections=allowed,local_decision=d,onset_context=case['context'],
            complete_dialogue=[{k:v for k,v in b.items() if k not in ('word_timing','word_timing_columns')}
                               for b in case.get('dialogue_context',case['context'])])
        for _ in range(2):
            body=dict(model=model,temperature=0,max_tokens=600,chat_template_kwargs={'enable_thinking':False},
                messages=[dict(role='system',content=RULES),dict(role='user',content=json.dumps(payload,separators=(',',':')))],
                response_format={'type':'json_schema','json_schema':dict(name='context_review',strict=True,schema=SCHEMA)})
            start=time.monotonic();attempt=dict(request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest());row['attempts'].append(attempt)
            try:
                req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=360) as f:r=json.load(f)
                choice=r['choices'][0];attempt.update(seconds=time.monotonic()-start,usage=r.get('usage'),finish_reason=choice['finish_reason'])
                if choice['finish_reason']!='stop':raise ValueError('unfinished_generation')
                answer=json.loads(choice['message']['content']);attempt['answer']=answer
                row['changed']=apply_review(c,d,answer,allowed);row['finding']=answer['finding'];break
            except Exception as e:attempt.update(error=str(e),seconds=time.monotonic()-start)
        if row.get('finding')=='NEW_FEEDBACK':
            # Repeated receipts to the exact same contribution extend one event.
            parent=feedback_parent(c,active_receipts)
            if parent:
                d.update(action='MERGE',merge_into=parent)
                row['finding']='CONTINUE_FEEDBACK'
            active_receipts[key]=(parent or c['candidate_id'],c['end'])
        row.update(after=copy.deepcopy(d),status='ok' if 'finding' in row else 'error_retained_original')
        reviews.append(row)
    result=finalize(case,candidates,judgment,speaker_gate(case,dict(mixed_audio_checked=True)))
    result.update(status='ok',candidates=candidates,judgment=judgment,context_reviews=reviews,
        attempts=[dict(a,stage='first') for a in baseline.get('attempts',[])]+[
            dict(a,stage='context_review',candidate_id=r['candidate_id']) for r in reviews for a in r['attempts']],
        conversation_id=case['conversation_id'],number=case['number'],target=case['target'],
        annotation_source='local_judge_with_grounded_context_review',context_policy=case.get('context_policy'),
        audio_semantic_judge_used=True,audio_view='mix',
        context_review_failed=any(r['status']!='ok' for r in reviews))
    return result
