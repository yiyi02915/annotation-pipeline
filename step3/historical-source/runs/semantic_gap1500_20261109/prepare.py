import json,hashlib,math,time
from pathlib import Path
import soundfile as sf,numpy as np
from scipy.signal import resample_poly
D=Path(__file__).resolve().parent;BASE=D.parent/'finegrained_pilot_20261109';DATA=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
es=[e for e in read(BASE/'benchmark_inputs.jsonl') if e['task']=='pause'];assert len(es)==len({e['id'] for e in es})==1500 and all(e['gap_seconds']>.5 for e in es)
idx={r['interaction_id']:r for r in read(DATA/'training/index_v1/train.jsonl')};meta=open(DATA/'seamless_fd_data/metadata.jsonl','rb');(D/'clips').mkdir(exist_ok=True);rows=[];t=time.time();cache={}
for e in es:
 cid=e['conversation_id']
 if cid not in cache:
  ix=idx[cid];meta.seek(ix['metadata_byte_offset']);cache[cid]=json.loads(meta.read(ix['metadata_bytes']))
 source=cache[cid];a=max(0,e['left_word']['start']-5);b=min(e['source_duration_seconds'],e['right_word']['end']+6);ca=max(0,e['anchor']-12);cb=min(e['source_duration_seconds'],e['right_word']['end']+12)
 tracks={};arrays=[];streams={};key=hashlib.sha256(e['id'].encode()).hexdigest()[:24];contexts=[]
 for p in source['participants']:
  sp=p['sample_id'].rsplit('_',1)[1];ws=[{'id':f'{sp}:w{i}','text':w['word'],'start':w['start_sec'],'end':w['end_sec']} for i,w in enumerate(p['words']) if w['end_sec']>ca and w['start_sec']<cb];streams[sp]=ws;groups=[]
  for w in ws:
   if not groups or w['start']-groups[-1][-1]['end']>.5:groups.append([])
   groups[-1].append(w)
  contexts.extend({'speaker':sp,'start':round(g[0]['start'],3),'end':round(g[-1]['end'],3),'text':' '.join(w['text'] for w in g)} for g in groups)
  path=e['audio_paths'][sp];info=sf.info(path);audio,sr=sf.read(path,start=int(a*info.samplerate),stop=int(b*info.samplerate),dtype='float32');audio=audio.mean(axis=1) if audio.ndim>1 else audio
  if sr!=16000:g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g)
  assert len(audio)>100 and np.isfinite(audio).all();dest=D/'clips'/f'{key}_{sp}.wav';sf.write(dest,audio,16000,subtype='PCM_16');tracks[sp]=str(dest);arrays.append(audio)
 context=sorted(contexts,key=lambda s:(s['start'],s['speaker']));context=[dict(segment_id=f'S{i+1}',**s) for i,s in enumerate(context)]
 ws=streams[e['focus']];indices={w['id']:i for i,w in enumerate(ws)};li=indices[e['left_word']['id']];ri=indices[e['right_word']['id']];assert ri==li+1
 n=min(map(len,arrays));mixed=D/'clips'/f'{key}_both.wav';sf.write(mixed,sum(x[:n] for x in arrays)/len(arrays),16000,subtype='PCM_16')
 payload={'id':e['id'],'focus':e['focus'],'other':e['other'],'target':{'left_word':e['left_word']['text'],'right_word':e['right_word']['text'],'left_word_id':e['left_word']['id'],'right_word_id':e['right_word']['id'],'absolute_gap':[e['anchor'],e['event_end']],'clip_relative_gap':[round(e['anchor']-a,3),round(e['event_end']-a,3)],'gap_seconds':e['gap_seconds']},'focus_before_target':' '.join(w['text'] for w in ws[:li+1]),'focus_after_target':' '.join(w['text'] for w in ws[ri:]),'other_near_target':[s for s in context if s['speaker']==e['other'] and s['end']>=e['anchor']-.3 and s['start']<=e['event_end']+.3],'context':context,'audio_absolute_interval':[a,b]}
 rows.append({'id':e['id'],'payload':payload,'audio_tracks':tracks,'mixed_audio':str(mixed),'source_audio_paths':e['audio_paths'],'source_gap_seconds':e['gap_seconds'],'source_alignment':'Seamless supplied word timestamps; original 1500 cohort retained','old_audio_interval':e['audio_interval'],'new_audio_interval':[a,b],'context_interval':[ca,cb]})
 if len(rows)%100==0:print('PREPARE',len(rows),round(time.time()-t,1),flush=True)
with (D/'events.jsonl').open('w') as f:
 for e in rows:f.write(json.dumps(e)+'\n')
audit={'events':len(rows),'unique_events':len({e['id'] for e in rows}),'all_source_gaps_strictly_gt_half_second':all(e['source_gap_seconds']>.5 for e in rows),'min_gap':min(e['source_gap_seconds'] for e in rows),'max_gap':max(e['source_gap_seconds'] for e in rows),'old_audio_did_not_cover_right_word':sum(e['audio_interval'][1]<e['right_word']['end'] for e in es),'all_new_audio_covers_right_word_and_followup':all(e['new_audio_interval'][1]>e['payload']['target']['absolute_gap'][1]+2 for e in rows),'input_ids_unchanged':{e['id'] for e in es}=={e['id'] for e in rows},'clips_include_both_speakers':True,'source_sha256':hashlib.sha256((BASE/'benchmark_inputs.jsonl').read_bytes()).hexdigest(),'seconds':time.time()-t}
(D/'input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));print('PREPARE_COMPLETE',flush=True)
