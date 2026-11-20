"""Typed interaction function and event continuity, then deterministic action mapping."""
import copy,hashlib,json,time,urllib.request
from pathlib import Path
from event_extractor import (extract_candidates,finalize,speaker_gate,refine_entry_decisions,
    suppress_terminal_answers,recover_echo_intervention,suppress_supported_floor_continuation)

RULES=Path(__file__).with_name('interaction_stateful.txt').read_text(encoding='utf-8')
EVENTS={'FEEDBACK':'BACKCHANNEL','FLOOR_CLAIM':'INTERRUPT'}
FUNCTIONS=list(EVENTS)+['ORDINARY_RESPONSE','OWN_CONTINUATION','SHARED_COMPLETION','NONE']
FIELDS={'candidate_id':{'type':'string'},'prior':{'type':'string','enum':['OPEN','FINAL_ANSWER_SLOT','ENDED','NONE']},
        'function':{'type':'string','enum':FUNCTIONS},'entry':{'type':'string','enum':['NEW','CONTINUATION']},
        'parent':{'type':'string'},'reason':{'type':'string'}}
SCHEMA={'type':'object','properties':{'states':{'type':'array','items':{'type':'object','properties':FIELDS,
    'required':list(FIELDS),'additionalProperties':False}}},'required':['states'],'additionalProperties':False}

def add_onset_features(candidates):
    byid={c['candidate_id']:c for c in candidates}
    for c in candidates:
        previous=byid.get(c.get('previous_own_candidate'))
        c['same_run_as_previous']=bool(previous and abs(c['speech_run_start']-previous['speech_run_start'])<1e-5)
        recipient=c.get('recipient_clause_at_onset')
        c['recipient_tail_seconds']=round(recipient['end']-c['start'],4) if recipient else None
        c['recipient_is_question']=bool(recipient and recipient['text'].rstrip().endswith(('?','？')))
        c['overlap_with_onset_clause']=round(sum(max(0,min(b,recipient['end'])-max(a,c['start']))
            for a,b in c['overlap_intervals']),4) if recipient else 0
    return candidates

def to_actions(candidates,states):
    byid={c['candidate_id']:c for c in candidates}
    if len(states)!=len(byid) or {s['candidate_id'] for s in states}!=set(byid):
        raise ValueError('states_must_cover_each_candidate_once')
    state={s['candidate_id']:s for s in states};decisions={}
    for c in sorted(candidates,key=lambda c:(c['start'],c['candidate_id'])):
        uid=c['candidate_id'];s=state[uid]
        if s['function'] not in FUNCTIONS or s['prior'] not in FIELDS['prior']['enum'] or s['entry'] not in FIELDS['entry']['enum']:
            raise ValueError('invalid_semantic_state')
        if s['function']=='FLOOR_CLAIM' and s['entry']=='NEW' and s['prior']!='OPEN':
            raise ValueError('floor_claim_requires_open_recipient:'+uid)
        label=EVENTS.get(s['function'])
        action='DROP';parent=''
        if label:
            if s['entry']=='NEW':
                if s['parent']:raise ValueError('new_event_has_parent:'+uid)
                action='KEEP'
            else:
                parent=s['parent'];p=byid.get(parent);pd=decisions.get(parent)
                if not (p and pd and p['start']<c['start'] and
                        (p['initiator'],p['recipient'])==(c['initiator'],c['recipient']) and
                        state[parent]['function']==s['function'] and pd['action'] in ('KEEP','MERGE')):
                    raise ValueError('invalid_semantic_continuation_parent:'+uid)
                action='MERGE'
        elif s['parent'] or s['entry']!='NEW':
            raise ValueError('non_event_cannot_claim_event_parent:'+uid)
        decisions[uid]=dict(candidate_id=uid,action=action,label=label or 'BACKCHANNEL',merge_into=parent,reason=s['reason'])
    return {'decisions':list(decisions.values())}

def classify(case,audio,*,endpoint='http://127.0.0.1:18770/v1/chat/completions',model='qwen38-27b'):
    candidates=add_onset_features(extract_candidates(case,refined=True,floor_checks=True))
    votes=audio.get('events',[]);vote_ids=[v['candidate_id'] for v in votes]
    required={c['candidate_id'] for c in candidates if c['other_active_run_at_onset'] and c['overlap_seconds']>0}
    if (audio.get('clip_id')!=case['id'] or len(vote_ids)!=len(set(vote_ids)) or
            not required<=set(vote_ids)<={c['candidate_id'] for c in candidates} or
            any(v['mix']['answer'] not in ('BACKCHANNEL','INTERRUPT','NEITHER') for v in votes)):
        raise ValueError('mixed_audio_votes_missing_or_invalid')
    vote_map={v['candidate_id']:v['mix'] for v in votes}
    aliases={c['candidate_id']:f'C{i}' for i,c in enumerate(candidates,1)}
    reverse={v:k for k,v in aliases.items()}
    rows=[]
    for c in candidates:
        v=vote_map.get(c['candidate_id'],{})
        c['mix_feedback_supported']=v.get('answer')=='BACKCHANNEL'
        c['audio_neither_probability']=v.get('probs',{}).get('NEITHER',0)
        row={k:v for k,v in c.items() if k not in ('word_ids','overlap_intervals','mix_feedback_supported','audio_neither_probability')}
        row['candidate_id']=aliases[c['candidate_id']]
        for key in ('previous_own_candidate','next_own_candidate'):
            if row.get(key):row[key]=aliases[row[key]]
        rows.append(row)
    payload=dict(target_interval=case['target'],candidates=rows,onset_context=case['context'],
        complete_dialogue=[{k:v for k,v in b.items() if k not in ('word_timing','word_timing_columns')}
                           for b in case.get('dialogue_context',case['context'])],
        context_policy=case.get('context_policy'),
        audio_votes=[dict(candidate_id=aliases[v['candidate_id']],label=v['mix']['answer']) for v in votes])
    gate=speaker_gate(case,dict(mixed_audio_checked=True));attempts=[]
    for _ in range(3):
        messages=[dict(role='system',content=RULES),dict(role='user',content=json.dumps(payload,separators=(',',':')))]
        if attempts:
            error=attempts[-1]['error']
            for uid,alias in aliases.items():error=error.replace(uid,alias)
            messages.append(dict(role='user',content='Previous output failed consistency validation: '+error+'. Return all candidate states, corrected.'))
        body=dict(model=model,messages=messages,temperature=0,max_tokens=4096,chat_template_kwargs={'enable_thinking':False},
            response_format={'type':'json_schema','json_schema':{'name':'semantic_event_states','strict':True,'schema':SCHEMA}})
        start=time.monotonic();attempt=dict(request_sent=True,request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest());attempts.append(attempt)
        try:
            req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=360) as stream:response=json.load(stream)
            choice=response['choices'][0];attempt.update(seconds=time.monotonic()-start,usage=response.get('usage'),finish_reason=choice['finish_reason'])
            if choice['finish_reason']!='stop':raise ValueError('unfinished_generation')
            states=json.loads(choice['message']['content'])['states']
            attempt['raw_states']=copy.deepcopy(states)
            for s in states:
                s['candidate_id']=reverse[s['candidate_id']]
                if s['parent']:s['parent']=reverse[s['parent']]
            raw=to_actions(candidates,states);judgment=copy.deepcopy(raw)
            repairs=refine_entry_decisions(candidates,judgment)
            repairs+=suppress_terminal_answers(candidates,judgment)
            repairs+=recover_echo_intervention(candidates,judgment)
            repairs+=suppress_supported_floor_continuation(candidates,judgment)
            result=finalize(case,candidates,judgment,gate)
            result.update(status='ok',candidates=candidates,judgment=judgment,semantic_states=states,
                pre_rule_judgment=raw,entry_repairs=repairs,attempts=attempts,number=case['number'],
                target=case['target'],conversation_id=case['conversation_id'],annotation_source='typed_semantics_plus_local_jev',
                audio_semantic_judge_used=True,audio_view='mix',context_policy=case.get('context_policy'))
            return result
        except Exception as exc:
            attempt.update(error=str(exc),seconds=time.monotonic()-start)
    return dict(status='error',attempts=attempts,conversation_id=case['conversation_id'],number=case['number'])
