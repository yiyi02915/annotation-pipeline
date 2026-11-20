"""Offline duplex annotations. Decisions are observations, never policy targets."""
from collections import defaultdict
import math

BOUNDARY_CODES={'A':'MERGE','B':'MERGE','C':'KEEP_BOUNDARY','D':'KEEP_BOUNDARY','E':'KEEP_BOUNDARY','F':'PENDING_AUTOMATIC'}
EVENT_TYPES={'BACKCHANNEL','INTERRUPT','TURN','OTHER','UNRESOLVED'}
FUNCTIONS={'ACKNOWLEDGE','STANCE','REPAIR','REDIRECT','OTHER','UNKNOWN'}
REACTIONS={'CONTINUE','YIELD','UNKNOWN'}
UPTAKE={'ADDRESSED','NOT_ADDRESSED','UNKNOWN'}

def compact(u):
    return {k:u[k] for k in ['utterance_id','speaker','start','end','text']}

def alignment_flags(u):
    flags=[]
    if len(u.get('words',[]))==1 and u['end']-u['start']>2:flags.append('single_word_over_2s')
    if any(isinstance(w.get('start'),(int,float)) and isinstance(w.get('end'),(int,float)) and w['end']-w['start']>3 for w in u.get('words',[])):
        flags.append('word_duration_over_3s')
    return flags

def context(record,start,end,before=30,after=10):
    return [compact(u) for u in sorted(record['utterances'],key=lambda u:(u['start'],u['end'],u['utterance_id'])) if u['end']>=start-before and u['start']<=end+after]

def make_boundaries(record):
    by_speaker=defaultdict(list)
    for u in record['utterances']:by_speaker[u['speaker']].append(u)
    out=[]
    for speaker,us in sorted(by_speaker.items()):
        us.sort(key=lambda u:(u['start'],u['end'],u['utterance_id']))
        for left,right in zip(us,us[1:]):
            flags=[]
            for side,u in [('left',left),('right',right)]:
                flags.extend(side+':'+flag for flag in alignment_flags(u))
            gap=round(right['start']-left['end'],3)
            if gap<0:flags.append('same_speaker_overlap')
            bid=record['conversation_id']+'::'+left['utterance_id']+'::'+right['utterance_id']
            out.append(dict(candidate_id=bid,conversation_id=record['conversation_id'],left=compact(left),right=compact(right),gap_seconds=gap,quality_flags=flags,context=context(record,left['start'],right['end'])))
    return out

def boundary_rule(c):
    if c['left']['speaker']!=c['right']['speaker']:return 'KEEP_BOUNDARY','different_speakers'
    if c['quality_flags']:return 'PENDING_AUTOMATIC','alignment_quality'
    if c['gap_seconds']>=3:return 'KEEP_BOUNDARY','gap_ge_3'
    return None

def validate_boundary(raw):
    if not isinstance(raw,dict) or raw.get('code') not in BOUNDARY_CODES:return None,'invalid_code'
    if not isinstance(raw.get('reason'),str) or len(raw['reason'].strip())<15:return None,'missing_evidence'
    return BOUNDARY_CODES[raw['code']],None

def make_events(record):
    """At most one directed event per incoming source utterance, overlap or gap."""
    us=sorted(record['utterances'],key=lambda u:(u['start'],u['end'],u['utterance_id']))
    out=[]
    for incoming in us:
        prior=[u for u in us if u['speaker']!=incoming['speaker'] and u['start']<incoming['start']]
        active=[u for u in prior if u['end']>incoming['start']]
        if active:floor=max(active,key=lambda u:u['start']);kind='overlap'
        elif prior:
            floor=max(prior,key=lambda u:u['end'])
            if incoming['start']-floor['end']>3:continue
            # Exclude repeated same-speaker continuation long after a completed exchange.
            newer_self=[u for u in us if u['speaker']==incoming['speaker'] and floor['end']<u['start']<incoming['start']]
            if newer_self:continue
            kind='gap'
        else:continue
        overlap=max(0,min(incoming['end'],floor['end'])-incoming['start'])
        flags=[side+':'+flag for side,value in [('floor',floor),('incoming',incoming)] for flag in alignment_flags(value)]
        out.append(dict(candidate_id=record['conversation_id']+'::EVENT::'+incoming['utterance_id'],conversation_id=record['conversation_id'],candidate_kind=kind,quality_flags=flags,floor=compact(floor),incoming=compact(incoming),overlap_seconds=round(overlap,3),gap_seconds=round(max(0,incoming['start']-floor['end']),3),context=context(record,min(floor['start'],incoming['start']),incoming['end'],before=15,after=15)))
    return out

def validate_event(c,raw):
    if not isinstance(raw,dict):return 'not_object'
    checks={'interaction_type':EVENT_TYPES,'function':FUNCTIONS,'floor_response':REACTIONS,'content_uptake':UPTAKE,'floor_completion':{'COMPLETE','INCOMPLETE','UNKNOWN'}}
    for key,allowed in checks.items():
        if raw.get(key) not in allowed:return 'invalid_'+key
    if len(str(raw.get('reason','')).strip())<15:return 'missing_evidence'
    if raw['interaction_type']=='INTERRUPT' and raw['floor_completion']!='INCOMPLETE':return 'interrupt_requires_ongoing_contribution_evidence'
    if raw['interaction_type']=='BACKCHANNEL' and raw['function'] in {'REPAIR','REDIRECT'}:return 'feedback_requires_response_conflict'
    secondary=raw.get('secondary_function','NONE')
    if secondary not in {'NONE','ACKNOWLEDGE','STANCE','REPAIR','REDIRECT','OTHER'} or secondary==raw['function']:return 'invalid_secondary_function'
    if raw['interaction_type']=='BACKCHANNEL' and secondary in {'REPAIR','REDIRECT'}:return 'secondary_feedback_conflict'
    for field,side in [('floor_quote','floor'),('incoming_quote','incoming')]:
        if not isinstance(raw.get(field),str) or not raw[field].strip() or raw[field] not in c[side]['text']:return 'ungrounded_'+field
    index=raw.get('response_context_index')
    if not isinstance(index,int) or isinstance(index,bool) or index < -1 or index>=len(c['context']):return 'invalid_response_index'
    if index>=0:
        target=c['context'][index]
        if target['speaker']!=c['floor']['speaker'] or target['start']<c['incoming']['start']:return 'invalid_response_chronology'
    if raw['content_uptake']=='ADDRESSED' and index==-1:return 'missing_uptake_evidence'
    # Do not confuse acoustic continuation with causal response. The LLM may say UNKNOWN.
    return None

def reconcile_boundary_evidence(records,boundaries,events):
    """An interruption of the right fragment cannot explain a boundary before it."""
    sources={u['utterance_id']:u for r in records for u in r['utterances']}
    issues=[]
    for b in boundaries:
        if (b.get('model_result') or {}).get('code')!='E':continue
        left=sources[b['left_id']];right=sources[b['right_id']]
        supporting=[e for e in events if e['status']=='accepted' and e['interaction_type']=='INTERRUPT'
                    and e['floor_response']=='YIELD' and e['floor_id']==b['left_id']
                    and left['start']<=e['start']<right['start']]
        b['supporting_interaction_ids']=[e['event_id'] for e in supporting]
        if not supporting:
            b.setdefault('decision_before_crosscheck',b['decision'])
            b['decision']='PENDING_AUTOMATIC'
            b['consistency_issue']='interruption_recovery_not_supported_at_this_boundary'
            issues.append(dict(boundary_id=b['candidate_id'],reason=b['consistency_issue'],human_required=False))
    return issues

def assemble_units(record,boundaries,events):
    decisions={(b['left_id'],b['right_id']):b for b in boundaries}
    incoming={e['incoming_id']:e for e in events if e['status']=='accepted'}
    by_speaker=defaultdict(list)
    for u in record['utterances']:by_speaker[u['speaker']].append(u)
    groups=[];overrides=[]
    for speaker,us in by_speaker.items():
        us.sort(key=lambda u:(u['start'],u['end'],u['utterance_id']));group=[]
        for u in us:
            merge=False
            if group:
                previous=group[-1];b=decisions.get((previous['utterance_id'],u['utterance_id']))
                merge=b is not None and b['decision']=='MERGE'
                e1=incoming.get(previous['utterance_id']);e2=incoming.get(u['utterance_id'])
                bc1=e1 is not None and e1['interaction_type']=='BACKCHANNEL'
                bc2=e2 is not None and e2['interaction_type']=='BACKCHANNEL'
                if merge and bc1!=bc2:
                    overrides.append(dict(boundary_id=b['candidate_id'],reason='backchannel_role_changed',from_decision='MERGE',to_decision='KEEP_BOUNDARY'));merge=False
                for e in events:
                    if merge and e['status']=='accepted' and e['interaction_type']=='INTERRUPT' and e['floor_response']=='YIELD' and e['floor_id']==previous['utterance_id'] and u['start']>=e['start']:
                        overrides.append(dict(boundary_id=b['candidate_id'],reason='interruption_recovery',event_id=e['event_id'],from_decision='MERGE',to_decision='KEEP_BOUNDARY'));merge=False
            if not merge and group:groups.append(group);group=[]
            group.append(u)
        if group:groups.append(group)
    groups.sort(key=lambda g:(g[0]['start'],g[0]['speaker']))
    units=[];mapping={}
    for n,g in enumerate(groups):
        uid=record['conversation_id']+'::U%04d'%n
        src=[u['utterance_id'] for u in g]
        role='BACKCHANNEL' if all(x in incoming and incoming[x]['interaction_type']=='BACKCHANNEL' for x in src) else 'MAIN'
        relevant=[b for b in boundaries if b['left_id'] in src or b['right_id'] in src]
        pending=any(b['decision']=='PENDING_AUTOMATIC' for b in relevant)
        unit=dict(unit_id=uid,speaker=g[0]['speaker'],start=g[0]['start'],end=g[-1]['end'],text=' '.join(u['text'] for u in g),source_utterance_ids=src,source_spans=[dict(utterance_id=u['utterance_id'],start=u['start'],end=u['end']) for u in g],role=role,boundary_status='PENDING_AUTOMATIC' if pending else 'accepted')
        unit['source_quality_flags']=[u['utterance_id']+':'+u['transcript_status'] for u in g if u.get('transcript_status') not in [None,'verified','verified_automatic']]
        unit['source_quality_flags'].extend(u['utterance_id']+':'+flag for u in g for flag in alignment_flags(u))
        unit['interaction_status']='PENDING_AUTOMATIC' if any(e['status']!='accepted' and (e['incoming_id'] in src or e['floor_id'] in src) for e in events) else 'accepted'
        units.append(unit);mapping.update({sid:uid for sid in src})
    return units,mapping,overrides

def control_timeline(record):
    """Observed speech activity from transcript timestamps; not VAD or desired policy."""
    us=record['utterances'];speakers=sorted({u['speaker'] for u in us})
    edges=sorted({round(u[k],6) for u in us for k in ['start','end']})
    frames=[];transitions=[];previous={s:False for s in speakers}
    for a,b in zip(edges,edges[1:]):
        active={s:any(u['speaker']==s and u['start']<b-1e-7 and u['end']>a+1e-7 for u in us) for s in speakers}
        frames.append(dict(start=a,end=b,acoustic_state='OVERLAP' if sum(active.values())>1 else 'SPEECH' if any(active.values()) else 'SILENCE',participants={s:'SPEAKING' if active[s] else 'NOT_SPEAKING' for s in speakers},basis='transcript_span_occupancy'))
        frames[-1]['input_quality_flags']=[u['utterance_id']+':'+flag for u in us if u['start']<b-1e-7 and u['end']>a+1e-7 for flag in alignment_flags(u)]
        for s in speakers:
            if active[s]!=previous[s]:transitions.append(dict(time=a,speaker=s,action='START_SPEAKING' if active[s] else 'STOP_SPEAKING',kind='observed_activity',policy_target=False))
        previous=active
    if edges:
        for s,v in previous.items():
            if v:transitions.append(dict(time=edges[-1],speaker=s,action='RECORDING_END',kind='observation_censored',policy_target=False))
    return frames,transitions

def validate_export(record,units,events,mapping):
    issues=[];expected={u['utterance_id'] for u in record['utterances']}
    actual=[sid for u in units for sid in u['source_utterance_ids']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:issues.append('source_coverage')
    for u in units:
        if u['end']<u['start']:issues.append('reversed_unit')
        source=[x for x in record['utterances'] if x['utterance_id'] in u['source_utterance_ids']]
        if len({x['speaker'] for x in source})!=1:issues.append('cross_speaker_merge')
    for e in events:
        if e['incoming_id'] not in mapping or e['floor_id'] not in mapping:issues.append('broken_event_link')
        if e['interaction_type']=='INTERRUPT' and e['status']=='accepted' and e['floor_completion']=='COMPLETE':issues.append('invalid_interrupt')
    return issues

def causal_text_context(record,cutoff,before=20):
    result=[]
    for u in sorted(record['utterances'],key=lambda u:(u['start'],u['end'])):
        if u['start']>=cutoff or u['end']<cutoff-before:continue
        if u['end']<=cutoff:result.append(compact(u));continue
        words=[w for w in u.get('words',[]) if isinstance(w.get('end'),(int,float)) and w['end']<=cutoff]
        if words:
            prefix=compact(u);prefix.update(text=' '.join(str(w.get('word',w.get('text',''))) for w in words),end=max(w['end'] for w in words),is_prefix=True)
            result.append(prefix)
    return result
