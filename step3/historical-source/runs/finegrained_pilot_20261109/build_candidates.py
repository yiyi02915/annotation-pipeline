"""Deterministic, label-blind event sampling from the official Seamless train split."""
import collections, hashlib, json, math, random, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
SEED=20261109
N=1500

def write(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))

def good(w):
    return (all(math.isfinite(w[k]) for k in ('start','end','score'))
            and w['start']>=0 and .015<=w['end']-w['start']<=2
            and w['score']>=.35 and bool(w['text'].strip()))

def runs(words):
    result=[]
    for w in words:
        if not result or round(w['start']-result[-1][-1]['end'],6)>.5:
            result.append([])
        result[-1].append(w)
    return result

def context(streams,a,b):
    out=[]
    for speaker,words in streams.items():
        selected=[w for w in words if w['end']>a and w['start']<b]
        for group in runs(selected):
            out.append({'speaker':speaker,'start':round(group[0]['start'],3),
                        'end':round(group[-1]['end'],3),'text':' '.join(w['text'] for w in group)})
    return sorted(out,key=lambda x:(x['start'],x['speaker']))

def union_length(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return sum(b-a for a,b in out)

def extract(row,index):
    if len(row['participants'])!=2:return []
    streams={};paths={};duration=min(p['speech']['duration_sec'] for p in row['participants'])
    for p,pi in zip(row['participants'],index['participants']):
        assert p['sample_id']==pi['sample_id']
        speaker=p['sample_id'].rsplit('_',1)[1]
        paths[speaker]=pi['audio_path'].replace('/hbt_share02/202610/','/data2/202610/')
        if not Path(paths[speaker]).is_file():return []
        streams[speaker]=[dict(id=f'{speaker}:w{i}',text=w['word'],start=float(w['start_sec']),end=float(w['end_sec']),score=float(w.get('score',1))) for i,w in enumerate(p['words'])]
        if any(b['start']<a['start']-.02 for a,b in zip(streams[speaker],streams[speaker][1:])):return []
    out=[];cid=row['interaction_id']
    for speaker,words in streams.items():
        other=next(s for s in streams if s!=speaker);ow=streams[other]
        for group in runs(words):
            start,end=group[0]['start'],group[-1]['end']
            if not all(good(w) for w in group) or end-start>12 or start<2 or end>duration-2:continue
            prior=[w for w in ow if w['start']<start-.04 and w['end']>start]
            if not prior or not all(good(w) for w in prior):continue
            intersections=[(max(x['start'],y['start']),min(x['end'],y['end'])) for x in group for y in ow if x['start']<y['end'] and y['start']<x['end']]
            overlap=union_length(intersections)
            if overlap<.08:continue
            a,b=max(0,start-8),min(duration,end+5)
            ctxwords=[w for ws in streams.values() for w in ws if w['end']>a and w['start']<b]
            if not ctxwords or sum(good(w) for w in ctxwords)/len(ctxwords)<.9:continue
            out.append(dict(id=f'{cid}:overlap:{group[0]["id"]}',task='overlap',focus=speaker,other=other,
                anchor=round(start,6),event_end=round(end,6),overlap_seconds=round(overlap,6),
                focus_text=' '.join(w['text'] for w in group),focus_word_ids=[w['id'] for w in group],
                context=context(streams,a,b),context_interval=[a,b],
                audio_interval=[max(0,start-.12),min(duration,end+.15)],
                alignment_min_score=min(w['score'] for w in group)))
        for left,right in zip(words,words[1:]):
            gap=round(right['start']-left['end'],6)
            if gap<=.5 or not good(left) or not good(right):continue
            if left['end']<3 or right['end']>duration-2:continue
            # Long gaps remain eligible: retain both ends and any intervening speaker words.
            a,b=max(0,left['end']-8),min(duration,right['end']+5)
            ctx=context(streams,a,b)
            if sum(len(x['text'].split()) for x in ctx)>600:continue
            ctxwords=[w for ws in streams.values() for w in ws if w['end']>a and w['start']<b]
            if not ctxwords or sum(good(w) for w in ctxwords)/len(ctxwords)<.9:continue
            out.append(dict(id=f'{cid}:pause:{left["id"]}:{right["id"]}',task='pause',focus=speaker,other=other,
                anchor=round(left['end'],6),event_end=round(right['start'],6),gap_seconds=gap,
                left_word=left,right_word=right,context=ctx,context_interval=[a,b],
                audio_interval=[max(0,left['end']-8),min(duration,left['end']+.5)],
                alignment_min_score=min(left['score'],right['score'])))
    for e in out:
        e.update(conversation_id=cid,session_id=cid.split('_I')[0],split=row['split'],conversation_kind=row['label'],
                 audio_paths=paths,source_duration_seconds=duration,
                 reference_label=None,reference_source=None,alignment_source='Seamless supplied word alignment',
                 context_mode='offline; includes subsequent speech',human_review_used=False)
    return out

def main():
    began=time.time();rng=random.Random(SEED)
    with (DATA/'training/index_v1/train.jsonl').open() as f:indices=[json.loads(line) for line in f]
    rng.shuffle(indices)
    pools={'overlap':[],'pause':[]};inspected=0;per_session=collections.Counter()
    with (DATA/'seamless_fd_data/metadata.jsonl').open('rb') as metadata:
        for ix in indices:
            session=ix['interaction_id'].split('_I')[0]
            if per_session[session]>=5:continue
            metadata.seek(ix['metadata_byte_offset']);row=json.loads(metadata.read(ix['metadata_bytes']))
            assert row['interaction_id']==ix['interaction_id']
            candidates=extract(row,ix);inspected+=1;per_session[session]+=1
            for task in pools:
                selected=[e for e in candidates if e['task']==task];rng.shuffle(selected)
                pools[task].extend(selected[:12])
            if inspected%20==0:
                progress=dict(stage='selecting',conversations_inspected=inspected,pools={k:len(v) for k,v in pools.items()})
                write('selection_progress.json',progress);print(json.dumps(progress),flush=True)
            if all(len(v)>=N+100 for v in pools.values()):break
    assert all(len(v)>=N for v in pools.values()),{k:len(v) for k,v in pools.items()}
    selected=[]
    for task,items in pools.items():
        rng.shuffle(items);selected.extend(items[:N])
    selected.sort(key=lambda e:(e['task'],e['conversation_id'],e['anchor']))
    with (ROOT/'candidate_reserve.jsonl').open('w') as f:
        used={e['id'] for e in selected}
        for items in pools.values():
            for e in items:
                if e['id'] not in used:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    assert len({e['id'] for e in selected})==2*N
    with (ROOT/'candidates.jsonl').open('w') as f:
        for e in selected:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    summary=dict(count=len(selected),by_task=dict(collections.Counter(e['task'] for e in selected)),
        unique_conversations=len({e['conversation_id'] for e in selected}),unique_sessions=len({e['session_id'] for e in selected}),
        by_kind=dict(collections.Counter(e['conversation_kind'] for e in selected)),split='train',seed=SEED,
        conversations_inspected=inspected,candidate_pool_sizes={k:len(v) for k,v in pools.items()},
        max_candidates_per_task_per_conversation=12,max_conversations_per_session=5,
        filters={'word_duration_seconds':[.015,2],'minimum_word_alignment_score':.35,'context_valid_word_fraction':.9,
                 'overlap_min_seconds':.08,'overlap_max_focus_seconds':12,'pause_min_seconds_exclusive':.5,
                 'max_context_words':600,'minimum_recording_margin_seconds':2},
        caveats=['Word-time overlap is not independently verified VAD overlap.','Alignment scores are upstream model scores, not calibrated probabilities.',
                 'Selected candidates are label-blind, not class-balanced.','No independent behavioral ground truth is available.'],
        manifest_sha256=hashlib.sha256((ROOT/'candidates.jsonl').read_bytes()).hexdigest(),elapsed_seconds=time.time()-began)
    write('selection_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
