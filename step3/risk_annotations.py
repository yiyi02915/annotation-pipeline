"""Keep a definite business label and a separate, auditable error-risk channel.

Risk is a rule-based indicator, never a calibrated probability of correctness.
"""
ALLOWED = {'gap': {'PAUSE', 'TURN_TRANSITION'},
           'overlap': {'INTERRUPT', 'BACKCHANNEL', 'NEITHER'}}


def annotate(task, initial_label, final_label, text_votes, audio_label,
             source_flags=(), rejudged=False):
    allowed = ALLOWED[task]
    if final_label not in allowed or initial_label not in allowed:
        raise ValueError('invalid_business_label')
    reasons = set(source_flags)
    if len(text_votes) < 2 or any(v not in allowed for v in text_votes):
        reasons.add('text_judge_failure')
    elif len(set(text_votes)) > 1:
        reasons.add('option_order_instability')
    if audio_label not in allowed:
        reasons.add('audio_judge_failure')
    elif any(v != audio_label for v in text_votes):
        reasons.add('text_audio_disagreement')
    if initial_label != final_label:
        reasons.add('label_changed_on_rejudge')
    stable = (len(text_votes) >= 2 and all(v == final_label for v in text_votes)
              and audio_label == final_label)
    unresolved = not stable or bool(set(source_flags))
    return dict(label=final_label, initial_label=initial_label,
                risk_flag=bool(reasons), risk_reasons=sorted(reasons),
                risk_unresolved=unresolved, rejudged=bool(rejudged),
                route='risk_subset' if unresolved else 'standard_subset',
                preference_eligible=not unresolved,
                risk_rule_version='v1_20261116',
                confidence_calibrated=False)


def tag_pipeline_result(result):
    """Attach quality metadata without changing labels or dropping flagged events."""
    risks=[]
    reviews={}
    review_scope={}
    timing_conflicts={}
    for w in result.get('windows',[]):
        for cid,interval in w.get('candidate_intervals',{}).items():
            core=w.get('target')
            if core is None or core[0]<=interval[0]<core[1]:
                review_scope[cid]=interval
        for audit in w.get('response_audits',[]):
            if audit.get('removal_blocked_reason'):
                timing_conflicts[audit['candidate_id']]=audit['removal_blocked_reason']
        for r in w.get('previous_context_reviews',[])+w.get('context_reviews',[]):
            old=reviews.get(r['candidate_id'],{})
            reviews[r['candidate_id']]=dict(r,changed=bool(old.get('changed') or r.get('changed')),
                status='error_retained_original' if 'error_retained_original' in (old.get('status'),r.get('status')) else r.get('status'))
    audio={v['candidate_id']:v.get('mix',{}) for w in result.get('windows',[]) for v in w.get('audio_votes',[])}
    for event in result.get('events',[]):
        flags=list(event.get('alignment_flags',[]))
        for cid in event.get('candidate_ids',[]):
            review=reviews.get(cid,{})
            if review.get('changed'):flags.append('label_changed_on_context_review')
            if review.get('status')=='error_retained_original':flags.append('context_review_failure')
            if cid in timing_conflicts:flags.append('semantic_review_timing_conflict')
            vote=audio.get(cid,{})
            if not vote.get('answer'): flags.append('audio_judge_missing')
            elif vote['answer']!=event['label']: flags.append('text_audio_disagreement')
            scores=sorted(vote.get('probs',{}).values(),reverse=True)
            if len(scores)>1 and scores[0]-scores[1]<.2: flags.append('small_audio_option_margin')
        flags=sorted(set(flags))
        event['quality']=dict(risk_flag=bool(flags),risk_reasons=flags,confidence_calibrated=False,risk_rule_version='pipeline_v1_20261116')
        if flags:risks.append(dict(kind='overlap',id=event['event_id'],label=event['label'],start=event['start'],end=event['end'],**event['quality']))
    for gap in result.get('gaps',[]):
        flags=[k for k,v in gap.get('risk',{}).items() if v]
        if gap.get('reviewed'):
            if not gap.get('review_complete'):flags.append('rejudge_failure')
            review=gap.get('review',{})
            labels=[review.get(k,{}).get('label') for k in ['short','verify']]
            if any(x is not None and x!=gap.get('label') for x in labels): flags.append('rejudge_label_disagreement')
        elif flags:flags.append('risk_not_yet_rejudged')
        if gap.get('label') is None:flags.append('no_valid_business_label')
        flags=sorted(set(flags))
        gap['quality']=dict(risk_flag=bool(flags),risk_reasons=flags,rejudged=gap.get('reviewed',False),confidence_calibrated=False,risk_rule_version='pipeline_v1_20261116')
        if flags:risks.append(dict(kind='gap',id=gap['id'],label=gap.get('label'),target=gap['target'],**gap['quality']))
    # Preserve evidence of possible omitted events, even when the text judge drops a candidate.
    seen=set()
    for window in result.get('windows',[]):
        for decision in window.get('decisions',[]):
            cid=decision.get('candidate_id')
            if cid in seen:continue
            interval=window.get('candidate_intervals',{}).get(cid)
            core=window.get('target')
            if interval is not None and core is not None and not core[0]<=interval[0]<core[1]:continue
            seen.add(cid)
            if decision.get('action')=='DROP' and audio.get(cid,{}).get('answer') in {'INTERRUPT','BACKCHANNEL'}:
                reasons=['possible_event_omission']
                if interval is None: reasons.append('candidate_scope_not_recorded')
                risks.append(dict(kind='rejected_candidate',id=cid,label=None,interval=interval,risk_flag=True,risk_reasons=reasons,audio_proposal=audio[cid]['answer'],confidence_calibrated=False))
    result['risk_items']=risks
    # Retain audit records for removed events and failed review, too.
    for cid,r in reviews.items():
        if cid not in review_scope:continue
        reasons=[]
        if r.get('changed'):reasons.append('label_changed_on_context_review')
        if r.get('status')=='error_retained_original':reasons.append('context_review_failure')
        if cid in timing_conflicts:reasons.append('semantic_review_timing_conflict')
        if not reasons:continue
        risks.append(dict(kind='context_review',id=cid,label=r.get('after',{}).get('label') if r.get('after',{}).get('action')!='DROP' else 'NEITHER',
                          interval=review_scope[cid],risk_flag=True,risk_reasons=reasons,confidence_calibrated=False))
    result['risk_summary']=dict(flagged_items=len(risks),flagged_annotations=sum(r['kind'] in ('gap','overlap') for r in risks),
        review_audit_items=sum(r['kind']=='context_review' for r in risks),
        possible_omissions=sum(r['kind']=='rejected_candidate' for r in risks),rule_version='pipeline_v2_20261117',not_calibrated=True)
    return result
