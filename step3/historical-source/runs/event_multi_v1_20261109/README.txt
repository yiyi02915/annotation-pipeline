多事件交互标注实验 v1

目标：一个片段包含多个有方向的交互事件。事件标签只有 BACKCHANNEL / INTERRUPT；KEEP / MERGE / DROP 是候选处理动作，不是第三种标签。

代码入口
1. prepare_server.py：按已选案例读取 Seamless 原始词级时间戳，保留双方流与上下文。
2. event_extractor.py：双向词级候选、时间交集、结构验证、方向约束、说话人数门槛。
3. interaction_multi_event_system.txt：整片段扫描、多事件、反向重入、嵌套反馈的判别提示词。
4. run_multi.py：调用本地 qwen38-27b 服务（18770），结构化输出，检查所有候选均有处理决定。
5. run_speaker_probe.py：JEV-4B 音频人数实验筛查。不是专用说话人模型，不能用作双人纯净性认证。
6. speaker_evidence.json：来自外部的第三说话人证据。此轮 O04 来源为用户截图反馈，不冒充自动检出。
7. compare.py：生成新旧对照，隔离排除片段，并导出诊断标注和训练标注。

事件字段
initiator / recipient：有方向的参与者ID。
start / end：由真实词时间戳锚定；跨短停顿合并同一介入时，保留连续事件范围。
overlap_intervals：实际词区间交集，可含多个区间；与整个事件持续时间不同。
candidate_ids / texts / reason：来源、原文与模型理由。
in_original_target：事件起点是否落在原黄色目标区间。该字段与整个上下文片段事件计数分开。

门槛语义
EXCLUDE：已知第三人，排除整片段。
UNVERIFIED：没有可信的声学人数核验；不输出训练条目。
PASS：外部可信声学证据通过，才可进入训练导出。
上述均为质量状态，不改变事件标签只有两类的约束。

结果文件
results.json：原始多事件重判及模型原始响应。
annotation_results.json：应用说话人门槛后的完整审计记录。
events.audit.jsonl：包括被排除片段的审计事件，不能直接用于训练。
events.jsonl：已去掉 EXCLUDE 片段的诊断事件；必须继续检查 training_eligible。
training_events.jsonl：仅 PASS 可进入，当前不得把空文件当成错误。
before_after.json / before_after.csv：逐片段对照。
speaker_probe.json：真实音频人数筛查输出，分数未经校准。

验证限制
新语义标签来自文本模型，未由独立金标确认；本轮音频模型只用于人数实验，不是语义事件的独立听音复核。
旧结果每条只保存一个目标事件，新结果覆盖整段上下文，计数增加不能全部解释为原算法漏标纠正。
现有第三说话人音频筛查漏检用户指出的 O04，不能报告为自动排除能力已验证。

本地结构回归测试：python test_events.py
本地生成对照：python compare.py（先取回真实 results.json / speaker_probe.json）
