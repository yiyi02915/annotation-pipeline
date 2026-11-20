"""Locate target words with contextual anchors; retain unresolved audio regions."""
import json,re,copy
from pathlib import Path
root=Path(__file__).resolve().parent
load=lambda n:json.loads((root/n).read_text(encoding='utf8'))
norm=lambda s:re.findall('[a-z0-9]+',s.lower().replace("'",'').replace('’',''))
def contains(full,part):return any(full[i:i+len(part)]==part for i in range(len(full)-len(part)+1))
records=load('records.initial_alignment.json');broad={t['utterance']['utterance_id']:t for t in load('alignment_evidence.json')}
probes=load('focused_evidence.json');decisions=[];regions=[]
def aligned(p,text):
    matches=[a for a in p['alignments'] if norm(a['text'])==norm(text) and a.get('words')]
    if len(matches)!=2 or len(matches[0]['words'])!=len(matches[1]['words']):return None
    a,b=matches[0]['words'],matches[1]['words']
    if norm(' '.join(w['word'] for w in a))!=norm(text):return None
    drift=max(abs(x[k]-y[k]) for x,y in zip(a,b) for k in ['start','end'])
    if drift>.15:return None
    words=[dict(word=x['word'],start=round((x['start']+y['start'])/2,3),end=round((x['end']+y['end'])/2,3)) for x,y in zip(a,b)]
    if any(not 0<=w['end']-w['start']<=2 for w in words):return None
    if words[0]['end']<=words[0]['start'] or words[-1]['end']<=words[-1]['start']:return None
    if any(y['start']-x['end']>.5 or y['start']<x['start'] for x,y in zip(words,words[1:])):return None
    return words,round(drift,3)
def repair(u,text,words,drift,old,probe):
    u.update(text=text,words=words,start=words[0]['start'],end=words[-1]['end'],transcript_status='verified_automatic')
    u['alignment_repair']=dict(method='contextual_asr_and_two_focused_alignments',gold_verified=False,original_source=copy.deepcopy(old),probe_id=probe,max_window_drift_seconds=drift,
        timing_scope='utterance_boundaries',zero_duration_internal_words=sum(w['end']==w['start'] for w in words))
for parent,t in broad.items():
    if t['utterance'].get('transcript_status'):continue
    ps=[p for p in probes if p['parent_id']==parent];old=t['utterance'];target=norm(old['text'])
    good=[]
    for p in ps:
        local=[norm(a['text']) for a in p['asr']];global_texts=[norm(a['text']) for a in t['asr']]
        supported=(all(contains(s,target) for s in local) or (any(contains(s,target) for s in local) and all(contains(s,target) for s in global_texts)))
        a=aligned(p,old['text']) if supported else None
        if a:good.append((p,a))
    if len(good)!=1:
        decisions.append(dict(parent_id=parent,accepted=False,reason='no_unique_stable_target_with_asr_support'));continue
    p,(words,drift)=good[0]
    if t['previous'] and words[0]['start']<t['previous']['end']-.03:
        decisions.append(dict(parent_id=parent,accepted=False,reason='target_overlaps_neighbor'));continue
    if t['next'] and words[-1]['end']>t['next']['start']+.03:
        decisions.append(dict(parent_id=parent,accepted=False,reason='target_overlaps_neighbor'));continue
    r=next(r for r in records if r['conversation_id']==t['conversation_id'])
    u=next(u for u in r['utterances'] if u['utterance_id']==parent)
    repair(u,old['text'],words,drift,old,p['probe_id'])
    decision=dict(parent_id=parent,accepted=True,target=dict(text=u['text'],start=u['start'],end=u['end']),extra_sources=[],unresolved_regions=[])
    used={p['probe_id']}
    for q in ps:
        if q['probe_id'] in used:continue
        texts=[x['text'] for x in q['asr']];text=texts[0]
        supported=bool(norm(text)) and norm(text)==norm(texts[1]) and all(contains(norm(a['text']),norm(text)) for a in t['asr'])
        aa=aligned(q,text) if supported else None
        if aa:
            ww,dd=aa
            # A newly discovered short contribution must remain distinct from the target.
            if ww[-1]['end']<=u['start'] or ww[0]['start']>=u['end']:
                sid=parent+'_audio_repair_'+str(len(decision['extra_sources']))
                extra=dict(utterance_id=sid,speaker=u['speaker'],source_parent_id=parent)
                repair(extra,text,ww,dd,old,q['probe_id']);r['utterances'].append(extra)
                decision['extra_sources'].append(dict(id=sid,text=text,start=extra['start'],end=extra['end']));used.add(q['probe_id'])
        if q['probe_id'] not in used:
            region=dict(region_id=q['probe_id'],conversation_id=r['conversation_id'],speaker=u['speaker'],start=q['hypothesized_span'][0],end=q['hypothesized_span'][1],reason='unresolved_local_transcription',asr_hypotheses=texts,human_required=False)
            regions.append(region);decision['unresolved_regions'].append(region['region_id'])
    decisions.append(decision)
for r in records:r['utterances'].sort(key=lambda u:(u['start'],u['end'],u['utterance_id']))
for n,obj in [('records.json',records),('focused_patches.json',decisions),('unresolved_audio_regions.json',regions)]:
    (root/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(decisions,ensure_ascii=False,indent=2))
