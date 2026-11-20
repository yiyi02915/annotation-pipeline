import json,collections,hashlib,zipfile
from pathlib import Path
from evidence import *
events=read(D/'adjudicated.jsonl');summary=json.loads((D/'summary.json').read_text());assert len(events)==291 and summary['qwen_completed']==291 and summary['jev_completed']==291
align_reasons={'alignment_window_unstable','gap_threshold_unstable','local_degenerate_anchor','degenerate_anchor','anchor_unmapped','local_anchor_unmapped_or_pending','poor_fa_lexical_mapping'}
asr_reasons={'anchor_text_not_confirmed_by_asr','asr_transcript_disagreement','audio_evidence_error','invalid_audio'}
groups=collections.Counter()
for e in events:
 if e['status']!='unresolved':continue
 r=set(e['reasons'])
 category='对齐或停顿边界不稳定' if r&align_reasons else '转录与音频证据不一致' if r&asr_reasons else '重判模型仍分歧' if 'models_disagree' in r else '音频模型对问法不稳定' if 'jev_question_unstable' in r else '模型判断不确定或依据不足'
 groups[category]+=1;e['primary_remaining_issue']=category
summary['remaining_primary_issue']=dict(groups)
summary['removed_after_two_fa_windows']=sum(e['status']=='no_longer_pause_candidate' for e in events)
summary['target_training_candidates']=sum(e['status']=='resolved_candidate' and e['final_label'] in ['HESITATION','TURN_TRANSITION'] for e in events)
summary['resolved_behavior_disagreements']=sum(e['status']=='resolved_candidate' for e in events)
summary['retained_unresolved']=sum(e['status']=='unresolved' for e in events)
summary['same_questions_consistent_not_independent_models']=True
summary['qwen_output_repair_count']=len(read(D/'qwen_output_repair.jsonl'))
summary['empty_track_rechecked']=len(read(D/'jev_rejudge_emptyfix.jsonl'))
repairs=read(D/'qwen_output_repair.jsonl')
summary['qwen_consistency_repair_label_changes']=sum(e.get('status')=='ok' and e.get('repair_of',{}).get('label')!=e['judgment']['label'] for e in repairs)
summary['qwen_consistency_repair_same_label']=sum(e.get('status')=='ok' and e.get('repair_of',{}).get('label')==e['judgment']['label'] for e in repairs)
with (D/'adjudicated.jsonl').open('w') as f:
 for e in events:f.write(json.dumps(e,ensure_ascii=False)+'\n')
for name,status in [('resolved_candidates','resolved_candidate'),('excluded_pause_candidates','no_longer_pause_candidate'),('unresolved','unresolved')]:
 with (D/f'{name}.jsonl').open('w') as f:
  for e in events:
   if e['status']==status:f.write(json.dumps(e,ensure_ascii=False)+'\n')
summary['audit']={'unique_ids':len({e['id'] for e in events}),'partition_total':sum(summary['status'].values()),'original_selection_rule':'Qwen HESITATION and JEV32 TURN_TRANSITION','passes':{'independent_asr_tracks':582,'expanded_fa_tracks':582,'focus_local_fa':291,'jev_both_speaker_audio_events':291,'qwen_context_events':291},'model_errors':sum(e['status']=='unresolved' and any('incomplete_or_error' in r for r in e['reasons']) for e in events)}
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
text=f'''291 条停顿分歧样本：自动检查与重判结果
本轮沿用原始 291 条事件 ID，不替换基线转录、时间或旧标签。

处理结果
- 获得一致且通过证据检查的行为标签：{summary['resolved_behavior_disagreements']} 条。
- 其中目标类别训练候选：{summary['target_training_candidates']} 条。
- 两种窗口复对齐均不再满足 >0.5 秒条件：{summary['removed_after_two_fa_windows']} 条，单独排除，不能当作行为标签正确数。
- 仍未解决：{summary['retained_unresolved']} 条，保留 UNCERTAIN，不依赖人工复核。
- 通过的类别分布：{summary['resolved_labels']}。

本轮做了什么
1. 从固定 3,000 条中选出 Qwen 判犹豫、JEV32 判话轮转换的全部 291 条。
2. 扩展到原停顿前 12 秒至后词结束后 8 秒；双方音轨分别进行 Qwen3-ASR-1.7B 识别和 Qwen3-ForcedAligner-0.6B 复对齐。
3. 对目标说话人再用停顿前后约 3 秒的局部窗口复对齐，比较两个窗口的词边界与 0.5 秒候选判定。
4. 保持真实同时关系，将双方音轨等增益混合；JEV32 对指定停顿重判，并用两种问法检查稳定性，每种平均 4 次选项排列。
5. Qwen3.8-27B 思考模式独立读取双人上下文、ASR 证据和复对齐时间，输出类别、前段状态、对方行为、后续承接和引用依据。不给模型旧标签。
6. 通过条件：音频/转录和边界检查通过；Qwen 与 JEV 一致；JEV 两种问法一致；引用有效且覆盖前后；类别与状态字段一致。OTHER 另存，UNCERTAIN 不进入确定标签训练集。
7. 对类别与状态字段矛盾、依据引用无效或输出异常的 Qwen 回答，以同一模型普通模式做自动一致性复判，允许修正类别而非强行修改字段；保留修正前后结果。这不是新增独立评审。
8. 修复空白音轨没有词可对齐时被误算成对齐失败的筛选问题；重新判断受影响的 50 条，不能用该筛选误报排除样本。

仍有的问题（互斥主因，按对齐→转录→模型分歧→稳定性→依据不足归类）
{json.dumps(dict(groups),ensure_ascii=False,indent=2)}

解释边界
“解决”仅指通过本轮自动证据和一致性规则，仍没有独立金标准，不能称为真实正确率。
两个音频问法来自同一个 JEV 模型，不是两个独立评审。Qwen 读取声音导出的识别和时间证据，JEV 直接读取双方混音；不声称 Qwen 本身直接听了音频。
0.2 秒窗口差、0.8 词映射比例和 0.6 转录相似度是实验筛选阈值，尚未经过准确率校准；ASR 不一致不直接证明原转录错。
本轮同时改变音频窗口、上下文、对齐、提示词和推理模式，不能把变化单独归因于某一项。
Control State 与全局语义合并切分不在本轮处理范围。
'''
(D/'结果说明.txt').write_text(text,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
