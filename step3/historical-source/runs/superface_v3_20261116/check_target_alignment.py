"""Check disputed target word anchors against existing source FA, never regenerate it blindly."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(ROOT/'step3'))
from input_seamless import load_published


def main():
    rows=json.loads((ROOT/'audit21_inputs.json').read_text())
    required={w.rsplit(':w',1)[0] for r in rows if not r['conversation_id'].startswith('V') for w in [r['input']['target'].get('left_word_id',''),r['input']['target'].get('right_word_id','')] if ':w' in w}
    chunks={}
    for path in (BASE/'ob141_smoketest_20261112/priority').rglob('predictions.jsonl'):
        for line in path.open():
            x=json.loads(line)
            if x.get('chunk_id') in required:chunks[x['chunk_id']]=x
        if required<=chunks.keys():break
    cids={r['conversation_id'] for r in rows if r['conversation_id'].startswith('V')}
    timelines={}
    for row,streams,tracks,error in load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),[dict(conversation_id=c) for c in cids]):
        if not error:timelines[row['conversation_id']]={w['id']:w for ws in streams.values() for w in ws}
    results=[]
    for r in rows:
        t=r['input']['target'];err=None;deltas=[]
        try:
            if r['task']=='gap':
                if r['conversation_id'].startswith('V'):
                    words=timelines[r['conversation_id']]
                    times=[words[t['left_word_id']]['end'],words[t['right_word_id']]['start']]
                else:
                    times=[]
                    for key,side in [('left_word_id',1),('right_word_id',0)]:
                        cid,idx=t[key].rsplit(':w',1);x=chunks[cid]
                        times.append(x['audio_start_sec']+x['force_align']['timestamps'][int(idx)][side])
                deltas=[a-b for a,b in zip(times,t['absolute_gap'])]
                if max(map(abs,deltas))>.08:err='target_alignment_mismatch'
                if times[1]-times[0]<=.5:err='source_gap_not_over_0_5s'
            else:
                wid=t['candidate_id'].split(':',1)[1].split('->')[0]
                w=timelines[r['conversation_id']][wid]
                deltas=[w['start']-t['start']]
                if abs(deltas[0])>.08:err='target_alignment_mismatch'
        except Exception as exc:err='source_fa_anchor_unavailable:'+type(exc).__name__
        results.append(dict(id=r['id'],verified=err is None,error=err,anchor_differences_seconds=deltas,note='timestamp consistency check, not independent phonetic re-alignment'))
    (ROOT/'alignment21.json').write_text(json.dumps(results,indent=2))
    lookup={r['id']:r for r in results}
    revised=json.loads((ROOT/'revised_train.json').read_text())
    for row in revised:
        a=lookup.get(row['id'])
        if a and not a['verified']:
            row['risk_reasons']=sorted(set(row['risk_reasons']+['source_fa_anchor_not_verified']))
            q=row['quality'];q.update(preference_eligible=False,risk_flag=True,risk_unresolved=True,route='risk_subset')
            q['risk_reasons']=sorted(set(q['risk_reasons']+['source_fa_anchor_not_verified']))
    (ROOT/'revised_train.json').write_text(json.dumps(revised,indent=2))
    print('ALIGNMENT',sum(r['verified'] for r in results),len(results),flush=True)


if __name__=='__main__':main()
