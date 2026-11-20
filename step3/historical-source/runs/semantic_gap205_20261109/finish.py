import json,collections,hashlib,zipfile
from pathlib import Path
from run import D,B,LABELS,events,payload,RULES,read
def write(p,rows):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
es=events();q={(x['id'],x['view']):x for k in range(3) for x in read(D/f'qwen_{k}.jsonl')};j={x['id']:x for x in read(D/'jev.jsonl')};arb={x['id']:x for x in read(D/'arbitration.jsonl')};selection=json.loads((D/'arbitration_selection.json').read_text());selected=set(selection['ids']);previous={x['id']:x for x in read(B/'adjudicated205.jsonl')};vad={x['id']:x for x in read(B/'vad_audit.jsonl')}
assert len(es)==len({e['id'] for e in es})==205
assert len(q)==410 and all(x['status']=='ok' for x in q.values())
assert len(j)==205 and all(x['status']=='ok' for x in j.values())
assert selected<=set(arb) and all(arb[k]['status']=='ok' for k in selected)
rows=[]
for e in es:
 ident=e['id'];a=q[ident,'a'];b=q[ident,'b'];ja=j[ident]['answers'];judged=arb[ident] if ident in selected else a;v=judged['judgment'];label=v['label'];assert label in LABELS
 votes={'text_context_first':a['judgment']['label'],'text_target_first':b['judgment']['label'],'audio_primary':ja['label']['answer'],'audio_rephrased':ja['check']['answer']};notes=[]
 if not e['boundary_stable']:notes.append('目标词边界未通过跨窗口稳定性检查')
 if abs(e['gap']-.5)<=.08:notes.append('重估时长接近0.5秒筛选门槛')
 if e['gap']<=.5:notes.append('重估时长不满足原始大于0.5秒筛选条件，仍按要求保留二分类标签')
 if vad.get(ident,{}).get('speech_inside_gap_conflict'):notes.append('目标空缺内检测到语音活动，需注意边界或转录误差')
 for flag in ['asr_transcript_disagreement','poor_fa_lexical_mapping','anchor_text_not_confirmed_by_asr']:
  if flag=='anchor_text_not_confirmed_by_asr' and e.get('alignment_repair',{} ) and e['alignment_repair'].get('accepted_boundary'):continue
  if flag in e['old_reasons']:notes.append(flag)
 if any(x!=label for x in votes.values()):notes.append('判断视角之间或与最终重判存在分歧，最终标签采用不展示旧答案的语义重判')
 if not v.get('references_valid'):notes.append('最终解释的上下文引用未通过自动检查')
 rows.append({'id':ident,'gap_label':label,'gap_label_zh':'停顿' if label=='PAUSE' else '话轮转换','definition_version':'binary_semantic_gap_v1','target':e['payload']['target'],'focus':e['payload']['focus'],'other':e['payload']['other'],'semantic_evidence':v,'final_decision_source':'fresh_semantic_arbitration' if ident in selected else 'four_views_agree','view_labels':votes,'views_supporting_final_label':sum(x==label for x in votes.values()),'audio_label_scores_not_calibrated':{k:ja[k]['probs'] for k in ['label','check']},'alignment_stable':e['boundary_stable'],'gap_seconds_reestimated':e['gap'],'quality_notes':notes,'previous_label':previous[ident]['final_label'],'previous_status':previous[ident]['status'],'audio_tracks':e['audio_tracks'],'mixed_audio':e['mixed_audio']})
write(D/'gap_labels.jsonl',rows);write(D/'labels_only.jsonl',[{'id':r['id'],'label':r['gap_label_zh']} for r in rows]);write(D/'model_inputs.jsonl',[payload(e) for e in es]);write(D/'comparison.jsonl',[{k:r[k] for k in ['id','previous_label','previous_status','gap_label','gap_label_zh','final_decision_source','view_labels','views_supporting_final_label','quality_notes']} for r in rows]);(D/'prompt_rules.txt').write_text(RULES,encoding='utf-8')
counts=collections.Counter(r['gap_label'] for r in rows);transition=collections.Counter((r['previous_label'],r['gap_label']) for r in rows);unanimous=sum(len(set(r['view_labels'].values()))==1 for r in rows);noquality=sum(not r['quality_notes'] for r in rows)
stats={'total':len(rows),'labels':dict(counts),'arbitrated':len(selected),'four_view_agreement':unanimous,'four_view_agreement_fraction':unanimous/len(rows),'no_recorded_quality_flags':noquality,'quality_notes_rows':205-noquality,'support_votes_distribution':dict(collections.Counter(r['views_supporting_final_label'] for r in rows)),'previous_to_new':[{'previous':a,'new':b,'count':n} for (a,b),n in sorted(transition.items())],'previously_unresolved_now_binary':sum(r['previous_status']=='unresolved' for r in rows),'alignment_unstable':sum(not r['alignment_stable'] for r in rows),'reestimated_gap_at_or_below_half_second':sum(r['gap_seconds_reestimated']<=.5 for r in rows),'qwen_requests_with_retry':sum(len(r['attempts'])>1 for r in list(q.values())+list(arb.values())),'note':'Every event receives one binary label. Complete label coverage is NOT validated accuracy. Multiple prompts of the same model are correlated; scores are not calibrated confidence. Quality metadata does not add a third class. Previous labels used a different taxonomy.'}
(D/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
audit={'unique_205':True,'exactly_two_allowed_labels':sorted(set(r['gap_label'] for r in rows)),'no_missing_labels':True,'text_views_complete_410':True,'audio_events_complete_205':True,'all_selected_arbitrations_completed':True,'source_events_sha256':sha(B/'events.jsonl'),'source_previous_results_sha256':sha(B/'adjudicated205.jsonl'),'prompt_source_sha256':sha(D/'run.py'),'transcript_and_audio_not_modified':True}
(D/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['205个目标空缺：按上下文语义进行二分类','',f'停顿：{counts["PAUSE"]}条；话轮转换：{counts["TURN_TRANSITION"]}条；合计205条。','没有OTHER、UNCERTAIN、拒判或多选标签。','只给原目标空缺打标，没有修改转录文本和音频。','','判别定义','停顿：后文继续、完成、补充、解释或修正停顿前仍在进行的同一个表达。该标签不额外声称心理上的犹豫。','话轮转换：前一表达已经结束，后文开始新的独立表达，或转而回应另一方新的实质发言。本轮采用语义贡献边界的操作性定义，不要求说话人ID一定改变。','只围绕同一个话题，不足以认定表达延续；前一句语法完整，也不足以认定表达已经结束。','','重跑方式','Qwen3.8-27B从完整上下文和目标空缺两个视角分别判断；JEV-32B结合原双方音频做两种语义问法判断。','只要四个判断不一致或引用不合格，就以原始上下文重新进行一次不展示旧答案的语义判定，并采用该二分类结果。','二次文本判定仍使用同一个Qwen模型，不视为独立金标准。',f'四个判断完全一致：{unanimous}条；进入语义重判：{len(selected)}条。',f'没有已记录质量标记：{noquality}条；其余{205-noquality}条保留质量信息，但同样只有一个二分类标签。','','质量信息与结果标签分开','对齐问题、时长门槛、音频活动疑点、模型分歧作为独立字段保存，不产生第三类。',f'其中，边界不稳定{stats["alignment_unstable"]}条；重估时长不超过0.5秒{stats["reestimated_gap_at_or_below_half_second"]}条。这些仍已输出二分类，但不能把完整覆盖等同于准确。','四种判断中两种来自同一个文本模型、两种来自同一个音频模型，不能视为四个独立模型；音频模型分数未做概率校准。','','与上一轮的比较','上一轮按更严格的证据门槛保留不确定；本轮按用户要求对全部205条必须二选一，且PAUSE不再要求额外的心理犹豫证据。','因此，原来未解决的样本获得二分类标签，不等于它们已经被独立验证正确。','旧标签 -> 新标签：']
for (a,b),n in sorted(transition.items()):lines.append(f'{a} -> {b}：{n}')
lines+=['','文件说明','labels_only.jsonl：主交付文件，只有样本ID和停顿/话轮转换二分类标签。','gap_labels.jsonl：附加审计文件，含语义解释、各视角判断和质量信息。','comparison.jsonl：与上一轮逐条对比。','model_inputs.jsonl / prompt_rules.txt：输入证据和本轮标准。','音频继续引用0493目录中上一轮保存的原始剪辑，本地不重复下载。']
(D/'结果说明.txt').write_text('\n'.join(lines),encoding='utf-8')
archive=D/'Seamless_205空缺语义二分类_20261109.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(D.iterdir()):
  if p.is_file() and p.suffix in ['.py','.json','.jsonl','.txt','.sh'] and p.name!='archive_sha256.txt':z.write(p,p.name)
(D/'archive_sha256.txt').write_text(sha(archive)+'  '+archive.name,encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2));print('FINAL_AUDIT_PASSED',flush=True)
