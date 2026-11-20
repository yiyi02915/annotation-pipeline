import json,sys,time,itertools,re,difflib
from pathlib import Path
import numpy as np,soundfile as sf,torch
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
D=Path(__file__).resolve().parent;B=D.parent/'rejudge291_20261109'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
def mapping(old,new):
 aa=[(c,i) for i,w in enumerate(old) for c in norm(w['text'])];bb=[(c,i) for i,w in enumerate(new) for c in norm(w['text'])];pairs={}
 for a,b,n in difflib.SequenceMatcher(None,[c for c,i in aa],[c for c,i in bb],autojunk=False).get_matching_blocks():
  for k in range(n):pairs.setdefault(aa[a+k][1],[]).append(bb[b+k][1])
 return {i:{'start':min(new[j]['start'] for j in js),'end':max(new[j]['end'] for j in js)} for i,js in pairs.items() if len(js)==len(norm(old[i]['text'])) and len(js)>0}
old={e['id']:e for e in read(B/'adjudicated.jsonl')};rows=[e for e in read(B/'inputs.jsonl') if old[e['id']].get('primary_remaining_issue')=='对齐或停顿边界不稳定'];assert len(rows)==86
asr=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=256)
fa=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
out=D/'alignment_repairs.jsonl';done={e['id'] for e in read(out)}
with out.open('a') as f:
 for e in rows:
  if e['id'] in done:continue
  t=time.time();ws=e['tracks'][e['focus']]['words'];idx={w['id']:i for i,w in enumerate(ws)};li=idx.get(e['left_word']['id']);ri=idx.get(e['right_word']['id']);r={'id':e['id'],'probes':[],'errors':[]}
  if li is None or ri is None:r['errors'].append('source_anchor_not_found')
  else:
   for count,pad in [(4,.25),(8,.45)]:
    words=ws[max(0,li-count):min(len(ws),ri+count+1)];a=max(e['expanded_interval'][0],words[0]['start']-pad);b=min(e['expanded_interval'][1],words[-1]['end']+pad)
    if b-a>25:a=max(a,e['anchor']-6);b=min(b,e['event_end']+6);words=[w for w in words if w['start']>=a and w['end']<=b]
    data,sr=sf.read(e['audio_paths'][e['focus']],start=int(a*sf.info(e['audio_paths'][e['focus']]).samplerate),stop=int(b*sf.info(e['audio_paths'][e['focus']]).samplerate),dtype='float32');data=data.mean(axis=1) if data.ndim>1 else data
    try:
     rec=asr.transcribe(audio=(data,sr),language='English')[0].text
     for mode,text in [('source',' '.join(w['text'] for w in words)),('asr',rec)]:
      if not text.strip():continue
      aligned=fa.align(audio=(data,sr),text=text,language='English')[0];aw=[{'text':w.text,'start':round(a+w.start_time,3),'end':round(a+w.end_time,3)} for w in aligned];mp=mapping(words,aw);ids={w['id']:i for i,w in enumerate(words)};left=mp.get(ids.get(e['left_word']['id']));right=mp.get(ids.get(e['right_word']['id']))
      valid=bool(left and right and .015<=left['end']-left['start']<=2 and .015<=right['end']-right['start']<=2 and left['start']<right['start'] and abs(left['end']-e['anchor'])<1 and abs(right['start']-e['event_end'])<1)
      r['probes'].append({'window':[a,b],'neighbor_words':count,'transcript_mode':mode,'asr':rec,'words':aw,'left':left,'right':right,'valid':valid,'gap':round(right['start']-left['end'],3) if left and right else None})
    except Exception as ex:r['errors'].append(str(ex))
   valid=[p for p in r['probes'] if p['valid']];clusters=[]
   for n in range(len(valid),1,-1):
    for cluster in itertools.combinations(valid,n):
     if len({p['neighbor_words'] for p in cluster})<2 or not any(p['transcript_mode']=='asr' for p in cluster):continue
     l=[p['left']['end'] for p in cluster];rr=[p['right']['start'] for p in cluster]
     if max(l)-min(l)<=.12 and max(rr)-min(rr)<=.12 and len({p['gap']>.5 for p in cluster})==1:clusters.append(cluster)
    if clusters:break
   if clusters:
    c=clusters[0];r['accepted_boundary']={'left_end':float(np.median([p['left']['end'] for p in c])),'right_start':float(np.median([p['right']['start'] for p in c])),'votes':len(c),'windows':len({p['neighbor_words'] for p in c}),'includes_asr_transcript':True};r['accepted_boundary']['gap']=round(r['accepted_boundary']['right_start']-r['accepted_boundary']['left_end'],3);r['repair_status']='stable_reestimated'
   else:r['repair_status']='still_unstable'
  r['seconds']=time.time()-t;f.write(json.dumps(r)+'\n');f.flush();done.add(e['id']);print('ALIGN',len(done),r.get('repair_status'),flush=True)
print('ALIGNMENT_COMPLETE',flush=True)
