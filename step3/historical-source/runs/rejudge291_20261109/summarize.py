import collections,json,time
from evidence import *
q={e['id']:e for e in read(D/'qwen_rejudge.jsonl')+[x for k in range(4) for x in read(D/f'qwen_rejudge_shard{k}.jsonl')]};j={e['id']:e for e in read(D/'jev_rejudge.jsonl')+read(D/'jev_rejudge_emptyfix.jsonl')};local={e['id']:e for e in read(D/'local_fa.jsonl')};events=[]
q.update({e['id']:e for e in read(D/'qwen_output_repair.jsonl') if e.get('status')=='ok'})
for e,p in ready():
 a=q.get(e['id'],{});b=j.get(e['id'],{});v=a.get('judgment',{});ans=b.get('answers',{});jl=ans.get('label',{}).get('answer');jc=ans.get('check',{}).get('answer');ql=v.get('label');reasons=list(p['quality_flags']);gap=p['alignment_diagnostics']['anchor']['new_gap']
 loc=local.get(e['id'],{});glob=p['alignment_diagnostics']['anchor'];local_difference=None
 if loc.get('left') and loc.get('right') and glob['left'] and glob['right']:
  local_difference=max(abs(loc['left']['end']-glob['left']['end']),abs(loc['right']['start']-glob['right']['start']))
  if local_difference>.2:reasons.append('alignment_window_unstable')
  if (loc['gap']>.5)!=(gap>.5):reasons.append('gap_threshold_unstable')
  if loc['left']['end']<=loc['left']['start'] or loc['right']['end']<=loc['right']['start']:reasons.append('local_degenerate_anchor')
 else:reasons.append('local_anchor_unmapped_or_pending')
 if a.get('status')!='ok':reasons.append('qwen_incomplete_or_error')
 if b.get('status')!='ok':reasons.append('jev_incomplete_or_error')
 if a.get('status')=='ok':
  if a.get('finish_reason')!='stop':reasons.append('qwen_output_truncated')
  if not v.get('evidence_ids_valid'):reasons.append('invalid_evidence_citation')
  refs={s['segment_id']:s for s in p['context']};used=[refs[x] for x in v.get('evidence_ids',[]) if x in refs];anchor=p['reestimated_event'] or p['original_event']
  if not any(s['start']<=anchor[0] for s in used) or not any(s['end']>=anchor[1] for s in used):reasons.append('missing_before_after_evidence')
  if ql=='HESITATION' and (v.get('left_state')!='unfinished' or v.get('focus_continuation')!='same_contribution'):reasons.append('label_evidence_conflict')
  if ql=='TURN_TRANSITION' and (v.get('left_state')!='completed_or_yielded' or v.get('focus_continuation')!='new_turn' or v.get('other_action')!='substantive_turn'):reasons.append('transition_evidence_insufficient')
 if ql!=jl:reasons.append('models_disagree')
 if jl!=jc:reasons.append('jev_question_unstable')
 if ql=='UNCERTAIN' or jl=='UNCERTAIN' or jc=='UNCERTAIN':reasons.append('model_uncertain')
 status='resolved_candidate' if not reasons else 'unresolved'
 if 'gap_no_longer_eligible' in reasons and not any(x in reasons for x in ['anchor_unmapped','anchor_text_not_confirmed_by_asr','degenerate_anchor','poor_fa_lexical_mapping','audio_evidence_error','asr_transcript_disagreement','alignment_window_unstable','gap_threshold_unstable','local_anchor_unmapped_or_pending','local_degenerate_anchor']):status='no_longer_pause_candidate'
 events.append({'id':e['id'],'status':status,'final_label':ql if status=='resolved_candidate' else 'UNCERTAIN','qwen_label':ql,'jev_label':jl,'jev_check':jc,'reasons':sorted(set(reasons)),'alignment':p['alignment_diagnostics'],'local_alignment':loc,'alignment_window_difference':local_difference,'quality_flags':p['quality_flags'],'qwen_judgment':v,'audio_interval':e['expanded_interval'],'audio_tracks':{sp:tr['audio'] for sp,tr in e['tracks'].items()},'mixed_audio':e['mixed_audio'],'evidence':p})
summary={'total_selected':291,'audio_checked':len(events),'qwen_completed':len(q),'jev_completed':len(j),'status':dict(collections.Counter(x['status'] for x in events)),'resolved_labels':dict(collections.Counter(x['final_label'] for x in events if x['status']=='resolved_candidate')),'qwen_labels':dict(collections.Counter(x['qwen_label'] for x in events)),'jev_labels':dict(collections.Counter(x['jev_label'] for x in events)),'remaining_reasons':dict(collections.Counter(r for x in events if x['status']=='unresolved' for r in x['reasons'])),'alignment_status':dict(collections.Counter(x['alignment']['anchor']['status'] for x in events)),'quality_flags':dict(collections.Counter(r for x in events for r in x['quality_flags'])),'new_model_agreement':sum(x['qwen_label']==x['jev_label'] and x['qwen_label'] is not None for x in events),'jev_question_stable':sum(x['jev_label']==x['jev_check'] and x['jev_label'] is not None for x in events),'accuracy_verified':False,'human_review_used':False,'notes':['Resolved means automatic evidence and consistency criteria pass, not verified accuracy.','FA is reestimated against original transcript, checked against independent ASR, not gold alignment.','0.2 s shift, 0.8 lexical mapping and 0.6 ASR similarity are experimental screening thresholds, not calibrated.','JEV hears both channels mixed in original time; separate-channel ASR and timing supply speaker attribution.','No original transcripts or baseline predictions are overwritten.']}
with (D/'adjudicated.jsonl').open('w') as f:
 for e in events:f.write(json.dumps(e,ensure_ascii=False)+'\n')
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
