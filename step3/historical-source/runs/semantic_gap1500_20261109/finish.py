import json,collections,hashlib,zipfile
from pathlib import Path
from run import D,R,read,events,ZH,RULES
def write(p,rows):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
es=events();q={x['id']:x for k in range(3) for x in read(D/f'qwen_{k}.jsonl')};j={x['id']:x for x in read(D/'jev.jsonl')};arb={x['id']:x for x in read(D/'arbitration.jsonl')};sel=json.loads((D/'arbitration_selection.json').read_text());selected={x['id'] for x in sel['selected']};ids={e['id'] for e in es}
assert len(es)==len(ids)==len(q)==len(j)==1500 and set(q)==set(j)==ids
assert all(e['source_gap_seconds']>.5 for e in es)
assert all(x['status']=='ok' for x in q.values()) and all(x['status']=='ok' for x in j.values())
assert selected<=set(arb) and all(arb[k]['status']=='ok' for k in selected)
old205={x['id']:x for x in read(D.parent/'semantic_gap205_20261109/gap_labels.jsonl')};old1500={x['id']:x for x in read(D.parent/'finegrained_pilot_20261109/qwen_predictions.jsonl')};known={x['id']:x for x in read(D.parent/'refine205_20261109/updated291.jsonl')};baseline291={x['id']:x for x in read(D.parent/'rejudge291_20261109/adjudicated.jsonl')};rows=[]
for e in es:
 ident=e['id'];a=q[ident];b=j[ident];ja=b['answers'];v=(arb[ident] if ident in selected else a)['judgment'];label=v['label'];assert label in ZH
 votes={'text_semantics':a['judgment']['label'],'audio_semantics':ja['label']['answer'],'audio_rephrased':ja['check']['answer']};notes=[]
 if any(x!=label for x in votes.values()):notes.append('初判存在分歧；采用不展示旧答案的语义重判结果')
 if not v.get('references_valid'):notes.append('最终解释的上下文引用未通过自动核验')
 k=known.get(ident,{});b291=baseline291.get(ident,{})
 if k.get('status')=='no_longer_pause_candidate':notes.append('历史重对齐提示不满足0.5秒门槛；本轮按原始时间戳固定样本重判语义，不能据此认定时间戳问题已修复')
 if k.get('boundary_stable') is False:notes.append('历史目标词边界稳定性检查未通过')
 if 'speech_detected_inside_target_gap' in k.get('reasons',[]):notes.append('历史检查在目标空缺内检测到语音活动')
 if k.get('status')=='unresolved' and 'boundary_stable' not in k and any(x in k.get('reasons',[]) for x in ['alignment_window_unstable','degenerate_anchor','local_degenerate_anchor','anchor_unmapped']):notes.append('历史对齐存在不稳定或退化结果')
 previous=old205.get(ident,{}).get('gap_label');r={'id':ident,'label':ZH[label],'label_code':label,'definition_version':'semantic_gap_binary_v2_same_speaker_new_topic','target':e['payload']['target'],'source_gap_seconds':e['source_gap_seconds'],'threshold_basis':'Seamless原始词时间戳，严格大于0.5秒；固定原1500条样本','semantic_evidence':v,'decision_source':'fresh_semantic_arbitration' if ident in selected else 'text_audio_two_forms_agree','view_labels':votes,'views_supporting_final':sum(x==label for x in votes.values()),'audio_scores_not_calibrated':{name:ja[name]['probs'] for name in ['label','check']},'quality_notes':notes,'previous205_label':previous,'old_baseline_qwen_label':old1500.get(ident,{}).get('label'),'audio_tracks':e['audio_tracks'],'mixed_audio':e['mixed_audio'],'audio_absolute_interval':e['new_audio_interval']};rows.append(r)
write(D/'labels_only.jsonl',[{'id':r['id'],'label':r['label']} for r in rows]);write(D/'gap_labels.jsonl',rows)
comparison=[{'id':r['id'],'previous_label':r['previous205_label'],'new_label':r['label_code'],'new_label_zh':r['label'],'changed':r['previous205_label']!=r['label_code']} for r in rows if r['previous205_label'] is not None];assert len(comparison)==205;write(D/'comparison_205.jsonl',comparison)
write(D/'quality_records.jsonl',[{'id':r['id'],'label':r['label'],'quality_notes':r['quality_notes'],'views_supporting_final':r['views_supporting_final']} for r in rows]);counts=collections.Counter(r['label'] for r in rows);unanimous=sum(len(set(r['view_labels'].values()))==1 for r in rows)
new_without_other=sum(r['label_code']=='TURN_TRANSITION' and not any(s['speaker']==e['payload']['other'] and s['end']>e['payload']['target']['absolute_gap'][0] and s['start']<e['payload']['target']['absolute_gap'][1] for s in e['payload']['context']) for r,e in zip(rows,es))
stats={'total':1500,'label_counts':dict(counts),'source_gaps_all_gt_half_second':True,'source_gap_range':[min(e['source_gap_seconds'] for e in es),max(e['source_gap_seconds'] for e in es)],'initial_three_judgments_agree':unanimous,'arbitration_count':len(selected),'final_label_support_distribution':dict(collections.Counter(r['views_supporting_final'] for r in rows)),'prior205_label_changes':sum(x['changed'] for x in comparison),'prior205_new_counts':dict(collections.Counter(x['new_label_zh'] for x in comparison)),'new_contribution_without_intervening_other_transcript':new_without_other,'known_reestimated_threshold_conflicts':sum(known.get(r['id'],{}).get('status')=='no_longer_pause_candidate' for r in rows),'with_quality_notes':sum(bool(r['quality_notes']) for r in rows),'no_recorded_quality_notes':sum(not r['quality_notes'] for r in rows),'primary_errors':sum(x['status']!='ok' for x in q.values()),'audio_errors':sum(x['status']!='ok' for x in j.values()),'arbitration_errors':sum(x['status']!='ok' for x in arb.values()),'note':'Two labels only. Agreement is not accuracy. Two audio question forms use one model; arbitration uses the same text model without old answers. Gap eligibility uses fixed original source timestamps; known revised boundary conflicts are preserved, not claimed fixed.'}
(D/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
audit={'exactly_1500_unique_ids':True,'exactly_two_allowed_label_strings':sorted(counts),'no_missing_labels':True,'all_primary_audio_and_selected_arbitration_outputs_present':True,'all_source_gaps_strictly_gt_half_second':True,'original_cohort_ids_unchanged':True,'all_audio_includes_post_gap_target_and_following_speech':True,'source_events_sha256':sha(D/'events.jsonl'),'main_labels_sha256':sha(D/'labels_only.jsonl'),'prompt_source_sha256':sha(D/'run.py'),'original_transcript_and_audio_not_edited':True};(D/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');(D/'prompt_rules.txt').write_text(RULES,encoding='utf-8')
lines=['Seamless 1500条空缺：语义二分类重跑','',f'同一发言内停顿：{counts["同一发言内停顿"]}条。',f'话轮交接间隔：{counts["话轮交接间隔"]}条。','1500条全部有唯一标签，没有第三类，没有空标签，没有待人工选择项。','','最终判据','同一发言内停顿：空缺后继续完成、补充、解释或修正空缺前的表达；另一方的简短反馈不改变这个关系。','话轮交接间隔：前一表达已结束/让出，后文开启新的独立表达、新话题，或回应另一方新的实质发言。按用户补充定义，同一个人可以开启新话题，不要求说话人变化，也不要求另一方必须先说话。','时间阈值只用于选候选，不用停顿长短替代语义判断。','','输入与音频修复','沿用原来的1500条停顿样本ID，按Seamless提供的原始词时间戳均严格大于0.5秒，范围0.501—45.652秒。','发现原1500条基线音频均未覆盖空缺后的目标词。本轮已全部重新读取双方原音频，保留目标词前约5秒、整个空缺及右侧目标词后约6秒；文本上下文扩展至前后约12秒。','长空缺的完整双方音频也予以保留；音频模型使用支持全长度的配置，未以停顿前的片段代替完整目标事件。','仅生成新的模型输入剪辑和标签，不修改原始转录/音频。','','判别方式','Qwen3.8-27B首先依据前后表达含义和语义关系判别；JEV-32B结合双方音频用两种问法复核。','初判不一致或证据引用不合格时，Qwen以原始上下文进行一次不展示旧答案的语义重判，最终仍必选一个标签。',f'初判三次判断一致：{unanimous}条；自动语义重判：{len(selected)}条。',f'上一轮205条子集有{sum(x["changed"] for x in comparison)}条标签变化，逐条对比见comparison_205.jsonl。','','质量与阈值说明',f'历史重对齐已提示{stats["known_reestimated_threshold_conflicts"]}条与0.5秒门槛存在冲突。本轮为了固定原1500条进行语义重跑，按原始时间戳保留这些样本，并单独记录该问题；这不代表时间戳已经修复，也不能把这些标签直接当作已核实大于0.5秒的训练样本。','模型分歧、对齐问题和证据记录与标签字段分开，不增加类别。','完整覆盖不等于准确率。未使用人工复核，也没有独立参考标注来测得真实准确率；模型分数没有做概率校准。','','交付文件','labels_only.jsonl：直接使用的主标签文件，每行只有id和label。','gap_labels.jsonl：逐条语义解释、判断来源、模型分数和音频路径。','quality_records.jsonl：单独的质量记录。','comparison_205.jsonl：与上一轮205条结果的对比。','输入音频位于服务器0493目录本次runs目录的clips下；本地归档不重复下载音频。']
lines+=['','本轮重跑的是语义标签和完整音频核验，没有对全部1500条重新执行独立词级对齐。没有既有质量记录的样本，不等于已经独立验证了边界或真实准确率。']
(D/'结果说明.txt').write_text('\n'.join(lines),encoding='utf-8')
manifest=[{'id':e['id'],'audio_paths':dict(e['audio_tracks'],both=e['mixed_audio']),'interval':e['new_audio_interval'],'samples_at_16000hz':round((e['new_audio_interval'][1]-e['new_audio_interval'][0])*16000)} for e in es];write(D/'audio_manifest.jsonl',manifest)
archive=D/'Seamless_1500空缺语义二分类_20261109.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(D.iterdir()):
  if p.is_file() and p.suffix in ['.py','.json','.jsonl','.txt','.sh'] and p.name!='archive_sha256.txt':z.write(p,p.name)
(D/'archive_sha256.txt').write_text(sha(archive)+'  '+archive.name,encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2));print('FINAL_AUDIT_PASSED',flush=True)
