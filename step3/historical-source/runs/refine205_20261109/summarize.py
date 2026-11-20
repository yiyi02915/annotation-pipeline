import json,collections,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'rejudge291_20261109'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def write(p,rows):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
old={r['id']:r for r in read(B/'adjudicated.jsonl')};events=read(D/'events.jsonl');q={r['id']:r for k in range(2) for r in read(D/f'qwen_{k}.jsonl')};q.update({r['id']:r for r in read(D/'qwen_repair.jsonl') if r['status']=='ok'});q.update({r['id']:r for r in read(D/'qwen_fact_clarification.jsonl') if r['status']=='ok'});j={r['id']:r for r in read(D/'jev.jsonl')};vad={r['id']:r for r in read(D/'vad_audit.jsonl')};rows=[]
for e in events:
 ident=e['id'];base=old[ident];qq=q.get(ident,{});jj=j.get(ident,{});v=qq.get('judgment',{});a=jj.get('answers',{});label=v.get('label','UNCERTAIN');jl=a.get('label',{}).get('answer');jc=a.get('check',{}).get('answer');jr=a.get('other_role',{}).get('answer');ir=jj.get('isolated_other',{}).get('other_role',{}).get('answer');reasons=[]
 if not e['boundary_stable']:reasons.append('target_boundary_unstable')
 if vad.get(ident,{}).get('speech_inside_gap_conflict'):reasons.append('speech_detected_inside_target_gap')
 if qq.get('label_changed') or qq.get('other_role_changed'):reasons.append('text_judgment_prompt_unstable')
 # Qwen FA timestamps are quantized; do not resolve threshold-edge cases by rounding.
 if e['boundary_stable'] and abs(e['gap']-.5)<=.08:reasons.append('gap_near_half_second_threshold')
 if qq.get('status')!='ok' or jj.get('status')!='ok' or qq.get('finish_reason')!='stop':reasons.append('model_output_failure_or_pending')
 if label=='UNCERTAIN' or jl=='UNCERTAIN':reasons.append('model_uncertain')
 if label!=jl:reasons.append('model_label_disagreement')
 if jl!=jc:reasons.append('audio_question_unstable')
 if v.get('other_action')!=jr or jr!=ir or jr=='unclear':reasons.append('other_function_unconfirmed')
 refs=v.get('evidence_ids',[]);segments={s['segment_id']:s for s in e['payload']['context']};cited=[segments[s] for s in refs if s in segments];l,r=e['payload']['target']['absolute_gap'];focus=e['payload']['focus']
 if not refs or any(s not in segments for s in refs) or not any(s['speaker']==focus and s['start']<=l+.2 for s in cited) or not any(s['speaker']==focus and s['end']>=r-.2 for s in cited):reasons.append('target_context_citation_missing')
 consistent=(label=='HESITATION' and v.get('focus_before_role')=='contributor' and v.get('left_state')=='unfinished' and v.get('focus_continuation')=='same_contribution' and v.get('pause_function')=='planning_or_repair' and v.get('other_action') in ['feedback_only','none']) or (label=='TURN_TRANSITION' and v.get('focus_before_role')=='contributor' and v.get('left_state')=='completed_or_yielded' and v.get('focus_continuation')=='new_turn' and v.get('pause_function')=='turn_exchange' and v.get('other_action')=='substantive_turn') or (label=='OTHER' and ((v.get('pause_function')=='listening_between_feedback' and v.get('focus_before_role')=='listener_feedback' and v.get('focus_continuation')=='listener_feedback' and v.get('other_action')=='substantive_turn') or (v.get('pause_function')=='breath_or_rhetorical' and v.get('focus_continuation')=='same_contribution') or (v.get('pause_function')=='interruption_suspension' and v.get('other_action')=='interrupting')))
 if not consistent and label!='UNCERTAIN':reasons.append('label_facts_conflict')
 # Local anchor repair cannot validate all remaining transcript content.
 for flag in ['asr_transcript_disagreement','poor_fa_lexical_mapping','invalid_audio','audio_evidence_error']:
  if flag in base.get('quality_flags',[]):reasons.append(flag)
 if not e.get('alignment_repair') and 'anchor_text_not_confirmed_by_asr' in base.get('quality_flags',[]):reasons.append('anchor_text_not_confirmed_by_asr')
 if e['boundary_stable'] and e['gap']<.42:status='no_longer_pause_candidate';final='NOT_ELIGIBLE';reasons=[]
 elif not reasons:status='resolved_candidate';final=label
 else:status='unresolved';final='UNCERTAIN'
 priority=['target_boundary_unstable','gap_near_half_second_threshold','speech_detected_inside_target_gap','model_output_failure_or_pending','asr_transcript_disagreement','poor_fa_lexical_mapping','model_label_disagreement','other_function_unconfirmed','audio_question_unstable','text_judgment_prompt_unstable','model_uncertain','label_facts_conflict','target_context_citation_missing']
 primary=next((x for x in priority if x in reasons),reasons[0] if reasons else None)
 rows.append({'id':ident,'prior_issue':e['prior_issue'],'status':status,'final_label':final,'primary_remaining_issue':primary,'reasons':reasons,'boundary_stable':e['boundary_stable'],'gap':e['gap'],'qwen_label':label,'jev_label':jl,'jev_check':jc,'qwen_other':v.get('other_action'),'jev_other_mixed':jr,'jev_other_isolated':ir,'qwen_judgment':v,'jev_answers':a,'jev_isolated_other':jj.get('isolated_other'),'target':e['payload']['target'],'audio_tracks':e['audio_tracks'],'mixed_audio':e['mixed_audio']})
 rows[-1]['vad_audit']=vad.get(ident);rows[-1]['text_field_clarification']=qq.get('repair');rows[-1]['text_label_changed']=qq.get('label_changed',False);rows[-1]['text_other_role_changed']=qq.get('other_role_changed',False)
write(D/'adjudicated205.jsonl',rows);byid={r['id']:r for r in rows};merged=[byid.get(k,v) for k,v in old.items()];write(D/'updated291.jsonl',merged)
write(D/'new_resolved.jsonl',[r for r in rows if r['status']=='resolved_candidate']);write(D/'new_excluded.jsonl',[r for r in rows if r['status']=='no_longer_pause_candidate']);write(D/'remaining205.jsonl',[r for r in rows if r['status']=='unresolved'])
stats={'total_targeted':len(events),'qwen_outputs':len(q),'jev_outputs':len(j),'by_original_group':{g:dict(collections.Counter(r['status'] for r in rows if r['prior_issue']==g)) for g in sorted(set(r['prior_issue'] for r in rows))},'new_status_counts':dict(collections.Counter(r['status'] for r in rows)),'new_resolved_labels':dict(collections.Counter(r['final_label'] for r in rows if r['status']=='resolved_candidate')),'boundary_repair86':dict(collections.Counter('stable' if e['boundary_stable'] else 'unstable' for e in events if e['alignment_repair'])),'remaining_primary_issues':dict(collections.Counter(r['primary_remaining_issue'] for r in rows if r['status']=='unresolved')),'remaining_overlapping_reasons':dict(collections.Counter(x for r in rows if r['status']=='unresolved' for x in r['reasons'])),'updated291_status_counts':dict(collections.Counter(r['status'] for r in merged)),'updated291_resolved_labels':dict(collections.Counter(r['final_label'] for r in merged if r['status']=='resolved_candidate')),'model_label_agreement':sum(r['qwen_label']==r['jev_label'] and r['qwen_label']!='UNCERTAIN' for r in rows),'other_function_three_way_agreement':sum(r['qwen_other']==r['jev_other_mixed']==r['jev_other_isolated'] and r['qwen_other'] not in ['unclear',None] for r in rows),'note':'Automatic evidence-screened candidates, not independently measured accuracy; no human review used. Gates are heuristic and not calibrated probabilities.'}
(D/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2))
