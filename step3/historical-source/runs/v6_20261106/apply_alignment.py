"""Deterministic acceptance of mutually consistent audio measurements, not gold truth."""
import json,re,copy
from pathlib import Path
root=Path(__file__).resolve().parent
normalize=lambda s:re.findall(r"[a-z0-9]+",s.lower().replace("'",'').replace('’',''))
def contains(full,part):return any(full[i:i+len(part)]==part for i in range(len(full)-len(part)+1))
records=json.loads((root/'records.baseline.json').read_text())
targets=json.loads((root/'alignment_evidence.json').read_text())
patches=[]
for t in targets:
    old=t['utterance'];texts=[a['text'] for a in t['asr']];source=normalize(old['text'])
    if len(texts)==2 and all(contains(normalize(s),source) for s in texts):
        text=old['text'];mode='original_text_supported'
    elif len(texts)==2 and normalize(texts[0])==normalize(texts[1]) and 0<len(normalize(texts[0]))<=max(8,len(source)*2):
        text=texts[0];mode='two_window_asr_text_revision'
    else:
        patches.append(dict(utterance_id=old['utterance_id'],accepted=False,reason='asr_windows_do_not_agree_on_target_text',asr_texts=texts));continue
    aligns=[a for a in t['alignments'] if a['text']==text and a.get('words')]
    if len(aligns)!=2 or len(aligns[0]['words'])!=len(aligns[1]['words']):
        patches.append(dict(utterance_id=old['utterance_id'],accepted=False,reason='incomplete_alignment_pair'));continue
    w0,w1=aligns[0]['words'],aligns[1]['words']
    drift=max(abs(a[k]-b[k]) for a,b in zip(w0,w1) for k in ['start','end'])
    new_words=[dict(word=a['word'],start=round((a['start']+b['start'])/2,3),end=round((a['end']+b['end'])/2,3)) for a,b in zip(w0,w1)]
    failures=[]
    if drift>.25:failures.append('window_alignment_drift_over_250ms')
    if any(not 0<w['end']-w['start']<=2 for w in new_words):failures.append('implausible_word_duration')
    if any(b['start']-a['end']>1.5 or b['start']<a['start'] for a,b in zip(new_words,new_words[1:])):failures.append('internal_timing_inconsistent')
    if normalize(' '.join(w['word'] for w in new_words))!=normalize(text):failures.append('forced_text_coverage_mismatch')
    if t['previous'] and new_words[0]['start']<t['previous']['end']-.1:failures.append('overlaps_previous_same_speaker')
    if t['next'] and new_words[-1]['end']>t['next']['start']+.1:failures.append('overlaps_next_same_speaker')
    item=dict(utterance_id=old['utterance_id'],accepted=not failures,reason=failures or [mode],asr_texts=texts,max_window_drift_seconds=round(drift,3),original=old,new_text=text,new_words=new_words)
    patches.append(item)
    if failures:continue
    u=next(u for r in records for u in r['utterances'] if u['utterance_id']==old['utterance_id'])
    u.update(text=text,words=new_words,start=new_words[0]['start'],end=new_words[-1]['end'],transcript_status='verified_automatic')
    u['alignment_repair']=dict(method='two_window_asr_and_forced_alignment',gold_verified=False,original_source=copy.deepcopy(old),max_window_drift_seconds=round(drift,3))
(root/'alignment_patches.json').write_text(json.dumps(patches,ensure_ascii=False,indent=2),encoding='utf8')
(root/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
(root/'records.initial_alignment.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps([dict(id=p['utterance_id'],accepted=p['accepted'],reason=p['reason'],new_text=p.get('new_text'),words=p.get('new_words')) for p in patches],ensure_ascii=False,indent=2))
