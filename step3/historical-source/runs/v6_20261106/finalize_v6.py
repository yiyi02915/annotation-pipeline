"""Audit, compare against the frozen baseline, and render an evidence report."""
import json,hashlib,html
from pathlib import Path
from collections import Counter
from pipeline_core import validate_export,alignment_flags
root=Path(__file__).resolve().parent;base=root/'baseline'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
rows=lambda p:[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
save=lambda n,v:(root/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
s=load(root/'summary.json');old_s=load(base/'summary.json')
records=load(root/'records.json');old_records=load(base/'records.json')
convs=rows(root/'annotated_conversations.jsonl');training=rows(root/'training_annotations.jsonl');pending=rows(root/'pending_automatic.jsonl')
old_training=rows(base/'training_annotations.jsonl');old_pending=rows(base/'pending_automatic.jsonl')
b=load(root/'boundary_results.json');e=load(root/'interaction_results.json');regions=load(root/'unresolved_audio_regions.json')
old_sources={u['utterance_id']:u for r in old_records for u in r['utterances']}
sources={u['utterance_id']:u for r in records for u in r['utterances']}
assert set(old_sources)<=set(sources) and len(sources)==sum(len(r['utterances']) for r in records)
for sid,u in sources.items():
    if sid not in old_sources:assert u['source_parent_id'] in old_sources
    if sid not in old_sources or u!=old_sources[sid]:
        repair=u['alignment_repair'];assert repair['original_source']==old_sources[u.get('source_parent_id',sid)]
        assert repair['max_window_drift_seconds']<=.25 and repair['gold_verified'] is False
        assert not alignment_flags(u)
    assert u['end']>=u['start']
for c in convs:
    record=next(r for r in records if r['conversation_id']==c['conversation_id'])
    mapping={sid:u['unit_id'] for u in c['semantic_units'] for sid in u['source_utterance_ids']}
    assert not validate_export(record,c['semantic_units'],c['interaction_events'],mapping)
    assert c['source_utterances']==record['utterances']
    for speaker in {u['speaker'] for u in record['utterances']}:
        ordered=sorted([u for u in record['utterances'] if u['speaker']==speaker],key=lambda u:u['start'])
        assert all(a['end']<=z['start'] for a,z in zip(ordered,ordered[1:]))
units={u['unit_id']:u for c in convs for u in c['semantic_units']}
for t in training:
    event=t['annotation'];assert event['status']=='accepted' and not event.get('audio_context_uncertainty')
    assert all(u['end']<=event['end']+1e-6 for u in t['input_context'])
    assert not t['policy_target']
    for key in ['floor_unit_id','incoming_unit_id']:
        u=units[event[key]];assert u['boundary_status']==u['interaction_status']=='accepted' and not u['source_quality_flags']
    assert not any(a['conversation_id']==event['conversation_id'] and a['start']<event['end']+15 and a['end']>event['start']-20 for a in regions)
assert all(p['human_required'] is False for p in pending)
assert len(training)==s['training_annotations'] and len(pending)==s['pending_items']
assert len({x['candidate_id'] for x in b})==len(b) and len({x['event_id'] for x in e})==len(e)
assert all(x.get('supporting_interaction_ids') for x in b if (x.get('model_result') or {}).get('code')=='E' and x['decision']!='PENDING_AUTOMATIC')
calls=rows(root/'model_calls.jsonl');hashes=set()
for call in calls:
    digest=lambda o:hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert digest(call['request'])==call['actual_request_hash'] and digest(call['initial_request'])==call['request_hash']
    assert call['model']=='glm-judge';hashes.add(call['request_hash'])
for x in b+e:
    if x.get('v6_origin')=='reused_unchanged_input':continue
    for j in x.get('judgments',[]):assert j['request_hash'] in hashes
audit=dict(integrity_passed=True,original_source_ids_preserved=len(old_sources),total_source_ids=len(sources),added_sources=len(set(sources)-set(old_sources)),
    changed_original_sources=[sid for sid in old_sources if sources[sid]!=old_sources[sid]],
    original_source_provenance_verified=True,no_same_speaker_overlap=True,training_inputs_no_future_text=True,
    unresolved_audio_context_excluded=True,model_request_hashes_verified=True,strong_model_calls=len(calls),
    model_output_parse_failures=sum(x['parsed'] is None for x in calls),human_review_used=False,accuracy_evaluated=False)
save('audit_results.json',audit)
old_ids={t['example_id'] for t in old_training};new_ids={t['example_id'] for t in training};event_index={x['event_id']:x for x in e};boundary_index={x['candidate_id']:x for x in b}
old_event_ids={x['event_id'] for x in load(base/'interaction_results.json')}
outcomes=[]
for p in old_pending:
    if p.get('task')=='semantic_boundary':
        cid=p['annotation']['candidate_id'];current=boundary_index.get(cid)
        outcomes.append(dict(id=cid,task='semantic_boundary',outcome=current['decision'] if current else 'CANDIDATE_REBUILT',reason=(current or {}).get('consistency_issue') or (current or {}).get('origin')))
    else:
        cid=p['example_id'];current=event_index.get(cid)
        outcome='EXPORTED' if cid in new_ids else 'CANDIDATE_REMOVED' if not current else 'STILL_EXCLUDED'
        record=next((x for x in pending if x.get('example_id')==cid),None)
        outcomes.append(dict(id=cid,task='interaction',outcome=outcome,reason=record.get('pending_reason') if record else None))
comparison=dict(baseline_training=len(old_ids),current_training=len(new_ids),newly_exportable=len(new_ids-old_ids),no_longer_exportable=len(old_ids-new_ids),net_gain=len(new_ids)-len(old_ids),
    added_training_ids=sorted(new_ids-old_ids),removed_training_ids=sorted(old_ids-new_ids),
    new_training_from_previous_candidates=len((new_ids-old_ids)&old_event_ids),new_training_from_new_candidates=len((new_ids-old_ids)-old_event_ids),
    previous_pending_outcomes=outcomes,previous_pending_outcome_counts=dict(Counter(x['outcome'] for x in outcomes)),
    old_candidate_count=len(old_event_ids),new_candidate_count=len(e),candidate_ids_added=sorted(set(event_index)-old_event_ids),candidate_ids_removed=sorted(old_event_ids-set(event_index)),
    remaining_boundary_reasons=dict(Counter(x.get('consistency_issue') or x['origin'] for x in b if x['decision']=='PENDING_AUTOMATIC')),
    remaining_event_validation_errors=dict(Counter(x['validation_error'] for x in e if x['validation_error'])),
    remaining_event_quality_flags=dict(Counter(flag for x in e for flag in x.get('input_quality_flags',[]))),
    remaining_audio_regions=regions,accuracy_evaluated=False)
old_events=load(base/'interaction_results.json');new_candidates={x['candidate_id']:x for x in load(root/'event_inputs.json')}
old_candidates={x['candidate_id']:x for x in load(base/'event_inputs.json')}
changed_ids=set(audit['changed_original_sources'])
comparison['alignment_affected_events']=[]
for old in old_events:
    if not {old['floor_id'],old['incoming_id']}&changed_ids:continue
    new=event_index.get(old['event_id']);candidate=new_candidates.get(old['event_id'])
    comparison['alignment_affected_events'].append(dict(id=old['event_id'],old_type=old['interaction_type'],new_type=new['interaction_type'] if new else 'CANDIDATE_REMOVED',old_kind=old['candidate_kind'],new_kind=candidate['candidate_kind'] if candidate else None,old_overlap=old['overlap_seconds'],new_overlap=candidate['overlap_seconds'] if candidate else None))
comparison['previous_invalid_event_outcomes']=[dict(id=old['event_id'],previous_error=old['validation_error'],new_status=event_index[old['event_id']]['status'] if old['event_id'] in event_index else 'CANDIDATE_REMOVED',new_error=event_index.get(old['event_id'],{}).get('validation_error'),new_type=event_index.get(old['event_id'],{}).get('interaction_type')) for old in old_events if old['validation_error']]
comparison['lost_training_reasons']=[dict(id=cid,pending_reason=next((x.get('pending_reason') for x in pending if x.get('example_id')==cid),'candidate_removed'),event_status=event_index.get(cid,{}).get('status')) for cid in sorted(old_ids-new_ids)]
comparison['remaining_unresolved_events']=[dict(id=x['event_id'],type=x['interaction_type'],reason=x['reason'],quality=x.get('input_quality_flags',[])) for x in e if x['status']!='accepted']
prior_four=load(base/'adjudicated_boundaries.json')
s['independent_dispute_results']={x['previous_candidate_id']:boundary_index.get(x['candidate_id'],{}).get('decision','CANDIDATE_REBUILT') for x in prior_four}
save('summary.json',s)
save('comparison.json',comparison)
escape=lambda x:html.escape(str(x))
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Seamless 自动修复与复判对比</title><style>body{font:16px/1.7 system-ui;color:#182a41;max-width:1180px;margin:32px auto;padding:0 24px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{border:1px solid #cbd5e1;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3fa}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}.note{color:#657489}</style><h1>Seamless：自动对齐修复与强模型复判</h1>'''
page+='<p>2026-11-06 · 音频多窗口识别与强制对齐 → 受影响边界重算 → GLM 独立复判 → 自动一致性校验 → 训练导出。未使用人工复核。</p>'
page+='<p><strong>训练导出：'+str(len(old_ids))+' → '+str(len(new_ids))+' 条；新增可用 '+str(len(new_ids-old_ids))+' 条，本轮排除原可用 '+str(len(old_ids-new_ids))+' 条，净增 '+str(len(new_ids)-len(old_ids))+' 条。</strong></p>'
page+='<table><tr><th>项目</th><th>上一轮</th><th>本轮</th></tr>'
for label,key in [('转录源片段','source_utterances'),('语义单元','semantic_units'),('交互候选','interaction_candidates'),('训练导出','training_annotations'),('自动待处理条目','pending_items')]:
    page+='<tr><td>'+label+'</td><td>'+escape(old_s[key])+'</td><td>'+escape(s[key])+'</td></tr>'
page+='</table><p class="note">可用指通过当前自动校验；没有独立金标，以上数字不是准确率。新增源片段保留父来源，候选重建、移除不算作标签修复成功。</p>'
page+='<h2>音频修复明细</h2><table><tr><th>原片段</th><th>旧时间与文本</th><th>当前时间与文本</th><th>结论</th></tr>'
for t in load(root/'alignment_evidence.json'):
    old=t['utterance'];u=sources[old['utterance_id']];changed=old!=u
    fmt=lambda q:'%.3f–%.3f 秒；%s'%(q['start'],q['end'],q['text'])
    page+='<tr><td>'+escape(old['utterance_id'])+'</td><td>'+escape(fmt(old))+'</td><td>'+escape(fmt(u))+'</td><td>'+('自动修复；原始值保留' if changed else '未修复，转录与对齐证据不足')+'</td></tr>'
page+='</table><p>另补回 '+str(len(set(sources)-set(old_sources)))+' 条在多窗口识别中一致、且时间稳定的短发言。自动验证采用同一 ASR 的不同窗口与独立强制对齐模型，不等于多个独立 ASR 或人工真值。修复主要用于句级边界，快速连读中仍可能有零长度词级时间。</p>'
if (root/'audio_evidence_manifest.json').exists():
    page+='<details><summary>原始音频证据片段（可选查看，不是流程中的人工复核步骤）</summary>'
    for a in load(root/'audio_evidence_manifest.json'):
        page+='<p>'+escape(a['parent_id'])+' · 原音频窗口 '+escape(a['source_window'])+' 秒</p><audio controls preload="none" src="'+escape(a['file'])+'"></audio>'
    page+='</details>'
page+='<h2>剩余问题</h2><table><tr><th>类别</th><th>当前结果</th></tr>'
for label,value in [('边界待处理原因',comparison['remaining_boundary_reasons']),('交互标签校验失败',comparison['remaining_event_validation_errors']),('交互输入质量标记',comparison['remaining_event_quality_flags']),('局部音频转录不确定区域',len(regions)),('因局部音频不确定而额外排除的训练条目',s['audio_context_excluded_training'])]:
    page+='<tr><td>'+label+'</td><td>'+escape(value)+'</td></tr>'
page+='</table><p>局部音频文字仍不确定时，其前后涉及输入或结果判断的上下文也不进入训练导出；这些条目继续保留在自动待处理队列。</p>'
page+='<h2>上一轮 49 个待处理条目的去向</h2><p>'+escape(comparison['previous_pending_outcome_counts'])+'</p><details><summary>展开逐项去向</summary><table><tr><th>编号</th><th>任务</th><th>本轮结果</th><th>原因</th></tr>'
for x in outcomes:page+='<tr>'+''.join('<td>'+escape(x.get(k,''))+'</td>' for k in ['id','task','outcome','reason'])+'</tr>'
page+='</table></details><h2>强模型判定与剩余依据</h2>'
for title,items in [('重新判定的边界',[x for x in b if x.get('v6_origin')=='glm_v6_boundary']),('重新判定的交互',[x for x in e if x.get('v6_origin')=='glm_v6_event'])]:
    page+='<details><summary>'+title+'（'+str(len(items))+'）</summary><pre>'+escape(json.dumps(items,ensure_ascii=False,indent=2))+'</pre></details>'
page+='<h2>验证范围</h2><p>来源与引用检查通过；保留全部原始来源，修订均有音频证据和原始值。训练输入不包含截点之后的文本。Control State 的活动区间仍来自转录占用，未替代真实 VAD；标注仍是观察性行为，不是理想行为策略。</p>'
page+='<details><summary>完整汇总与审计</summary><pre>'+escape(json.dumps(dict(summary=s,audit=audit),ensure_ascii=False,indent=2))+'</pre></details></html>'
(root/'自动修复与复判对比.html').write_text(page,encoding='utf8')
print(json.dumps(dict(summary=s,audit=audit,comparison={k:v for k,v in comparison.items() if k not in ['previous_pending_outcomes','added_training_ids','removed_training_ids']}),ensure_ascii=False,indent=2))
