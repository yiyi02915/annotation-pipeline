import json,hashlib,collections,zipfile,time,re
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'rejudge291_20261109'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
events=read(D/'events.jsonl');rows=read(D/'adjudicated205.jsonl');merged=read(D/'updated291.jsonl');s=json.loads((D/'summary.json').read_text());rep=read(D/'alignment_repairs.jsonl');jev=read(D/'jev.jsonl');qraw=[r for k in range(2) for r in read(D/f'qwen_{k}.jsonl')];clar=read(D/'qwen_fact_clarification.jsonl');recovered=read(D/'qwen_repair.jsonl');old={r['id']:r for r in read(B/'adjudicated.jsonl')};ids={r['id'] for r in events}
assert len(ids)==len(events)==len(rows)==205
assert len({r['id'] for r in qraw})==len(qraw)==len(jev)==205
assert len({r['id'] for r in merged})==len(merged)==291
assert all(r['status']=='ok' for r in jev)
assert len(rep)==86
assert all(old[r['id']]['status']=='unresolved' for r in rows)
assert all(old[r['id']]==r for r in merged if r['id'] not in ids)
assert all(not r['reasons'] and r['qwen_label']==r['jev_label']==r['jev_check']==r['final_label'] and r['qwen_other']==r['jev_other_mixed']==r['jev_other_isolated'] for r in rows if r['status']=='resolved_candidate')
assert all(r['boundary_stable'] and r['gap']<.42 for r in rows if r['status']=='no_longer_pause_candidate')
assert not any('model_output_failure_or_pending' in r['reasons'] for r in rows)
gap_probes=read(D/'gap_audio_probe.jsonl');assert len(gap_probes)==36 and all(r['status']=='ok' for r in gap_probes)
event_map={e['id']:e for e in events};gap_diagnostics=collections.Counter()
for g in gap_probes:
 target=event_map[g['id']]['payload']['target'];norm=lambda text:re.sub('[^a-z0-9]','',text.lower());anchors={norm(target['left_word']),norm(target['right_word'])};inside=[w for w in g['words_inside_marked_gap'] if .04<=w['end']-w['start']<=2]
 kind='additional_word_hypothesis' if any(norm(w['text']) not in anchors for w in inside) else ('anchor_extends_into_gap_hypothesis' if inside else 'no_stable_lexical_explanation')
 gap_diagnostics[kind]+=1
audit={'unique_target205':True,'complete_baseline291':True,'all_original_qwen_and_audio_outputs_present':True,'all_unaffected86_records_unchanged':True,'all_new_resolved_pass_gates':True,'all_new_excluded_safely_below_threshold':True,'technical_failures_remaining':0,'baseline_sha256':sha(B/'adjudicated.jsonl'),'inputs_sha256':sha(B/'inputs.jsonl'),'events_sha256':sha(D/'events.jsonl'),'unresolved_not_forced_to_training':True}
(D/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
new=collections.Counter(r['final_label'] for r in rows if r['status']=='resolved_candidate');closed=sum(r['status']!='unresolved' for r in rows);remain=205-closed
new_target=new['HESITATION']+new['TURN_TRANSITION'];all_target=sum(r['status']=='resolved_candidate' and r['final_label'] in ['HESITATION','TURN_TRANSITION'] for r in merged)
names={'target_boundary_unstable':'目标词/停顿边界仍不稳定','gap_near_half_second_threshold':'停顿紧邻0.5秒门槛，暂不据此纳入或剔除','speech_detected_inside_target_gap':'目标停顿内部仍检测到语音活动','model_label_disagreement':'文本与音频模型标签仍不同','other_function_unconfirmed':'另一方发言功能尚未交叉确认','audio_question_unstable':'音频模型换问法后不稳定','text_judgment_prompt_unstable':'澄清字段后文本标签或另一方功能变化','model_uncertain':'模型证据不足','label_facts_conflict':'结构化事实与最终标签仍矛盾','asr_transcript_disagreement':'转录与重新识别仍不一致','poor_fa_lexical_mapping':'上下文词级映射仍不可靠','target_context_citation_missing':'缺少目标前后有效引用'}
lines=['Seamless 205条分歧样本自动修复与重判结果','',f'本轮处理205条：原对齐问题86条，原模型分歧119条。自动闭环{closed}条，仍保留不确定{remain}条。',f'新增可接纳候选{sum(new.values())}条：犹豫{new["HESITATION"]}、话轮转换{new["TURN_TRANSITION"]}、其他行为{new["OTHER"]}。',f'其中可用于当前犹豫/话轮转换二分类的新增候选为{new_target}条；OTHER单独保存，不计入二分类训练量。',f'新增排除停顿候选{s["new_status_counts"].get("no_longer_pause_candidate",0)}条，均有稳定重估边界且低于0.42秒；不是行为分类正确率。','','一、分组结果']
for group,counts in s['by_original_group'].items():lines.append(f'{group}：新增候选{counts.get("resolved_candidate",0)}，排除{counts.get("no_longer_pause_candidate",0)}，仍未解决{counts.get("unresolved",0)}。')
lines+=['','二、对齐修复','86条各使用两组邻近词窗口，分别对原转录与重新ASR文本强制对齐；共最多4组证据。','精确匹配完整目标词，拒绝把短词匹配到其他单词内部；目标边界需跨窗口相差不超过0.12秒，包含ASR文本对齐，且其他有效结果不能存在大于0.2秒或门槛两侧的冲突。','26条通过上述词边界稳定性检查，60条未通过。26条中重估停顿<=0.5秒有5条；其中2条为0.48秒，因接近门槛仍保留不确定，只有3条安全排除。','边界稳定不等于行为标签可靠，仍须通过语音活动、语义、另一方功能与模型稳定性检查。','对205条另做Silero VAD双阈值（0.35/0.65）检查。停顿内部两次都检测到超过0.2秒语音活动时保留不确定；VAD本身不能证明词义，也可能受短声、串音等影响。','','三、目标定位与另一方功能','每条明确给出左右目标词、词ID、绝对时间和剪辑内相对时间；保留较长文本上下文，另裁剪目标前后约4秒的双方同步音频。','JEV-32B对混合音频做两种问法的目标标签判断，另判另一方功能；再用另一方独立声道复查该功能。每题采用4种选项排列。','Qwen3.8-27B盲判目标上下文，输出前后参与角色、发言是否持续、另一方功能、停顿功能与证据引用，不提供旧标签。','发现“语法完整”被误解为“发言结束”后，澄清字段定义并重新盲判字段矛盾项；该步骤仍是同一个模型，不计为独立投票。若标签或另一方功能改变，则保留不确定。','接纳要求：目标边界可靠、远离0.5秒门槛、无明显语音/转录冲突、文本与音频标签一致、音频两问法一致、另一方功能在文本/混合音频/独立声道三者一致、字段与标签一致、证据引用有效。','','四、仍未解决的主要问题（互斥计数，仅本轮205条）']
for issue,count in s['remaining_primary_issues'].items():lines.append(f'{names.get(issue,issue)}：{count}条。')
lines+=['','五、回填原291条后的进度',f'可接纳候选：22 -> {s["updated291_status_counts"].get("resolved_candidate",0)}。',f'排除停顿候选：36 -> {s["updated291_status_counts"].get("no_longer_pause_candidate",0)}。',f'仍未解决：233 -> {s["updated291_status_counts"].get("unresolved",0)}。',f'当前犹豫/话轮转换二分类训练候选：21 -> {all_target}。','原来未包含在本轮205条中的86条记录保持原样；其中原已解决/排除58条，原未解决28条。本轮新增门槛检查未追溯重跑这些记录。','','六、计数与置信度边界','这些数字表示自动证据筛选后的候选量，不是独立金标准准确率。模型置信分数与多次一致性均未做概率校准。','不依赖人工复核；未解决样本自动保留为UNCERTAIN并隔离，不强制分为犹豫或话轮转换。','本轮只处理停顿任务，没有新增interrupt/backchannel独立评测；另一方反馈功能是本轮辅助判据。','音频保留于服务器/data/L202500493/duplex_postprocess/runs/refine205_20261109/clips；本地归档包含结果、证据、配置和重跑代码，未重复下载音频。']
lines+=['','七、36条停顿内语音活动疑点的进一步诊断',f'局部ASR/对齐出现额外词线索：{gap_diagnostics["additional_word_hypothesis"]}条；主要是锚词伸入停顿的线索：{gap_diagnostics["anchor_extends_into_gap_hypothesis"]}条；尚无稳定词级解释：{gap_diagnostics["no_stable_lexical_explanation"]}条。','这些只是自动诊断线索，可能包含漏词、边界偏移、短反馈或声道串音；不据此强行接纳或改标签。']
(D/'结果说明.txt').write_text('\n'.join(lines),encoding='utf-8')
extra={'new_binary_training_candidates':new_target,'updated_binary_training_candidates':all_target,'original_qwen_error_or_truncation':sum(r['status']!='ok' or r.get('finish_reason')!='stop' for r in qraw),'recovered_outputs':len(recovered),'fact_clarifications':len(clar),'clarification_label_changes':sum(r.get('label_changed',False) for r in clar),'clarification_role_changes':sum(r.get('other_role_changed',False) for r in clar),'jev_question_stable':sum(r['answers']['label']['answer']==r['answers']['check']['answer'] for r in jev),'jev_other_role_stable':sum(r['answers']['other_role']['answer']==r['isolated_other']['other_role']['answer'] for r in jev),'jev_inference_seconds':sum(r['seconds'] for r in jev),'cumulative_qwen_request_seconds_not_walltime':sum(r['seconds'] for r in qraw+clar+recovered),'alignment_compute_seconds':sum(r.get('seconds',0) for r in rep)}
extra['gap_audio_diagnostic_hypotheses']=dict(gap_diagnostics)
(D/'additional_statistics.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
manifest=[]
for e in events:
 paths=dict(e['audio_tracks'],both=e['mixed_audio']);manifest.append({'id':e['id'],'audio':{k:{'path':v,'sha256':sha(Path(v)),'bytes':Path(v).stat().st_size} for k,v in paths.items()}})
(D/'audio_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in manifest),encoding='utf-8')
archive=D/'Seamless_205自动修复重判_20261109.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(D.iterdir()):
  if p.is_file() and p.suffix in ['.py','.json','.jsonl','.txt','.sh'] and p.name!='archive_sha256.txt':z.write(p,p.name)
(D/'archive_sha256.txt').write_text(sha(archive)+'  '+archive.name,encoding='utf-8')
print(json.dumps(extra,ensure_ascii=False));print('FINAL_AUDIT_PASSED',archive)
