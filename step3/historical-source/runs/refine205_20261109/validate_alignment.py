import json,re,difflib,itertools,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent/'rejudge291_20261109'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
def mapping(old,new):
 aa=[(c,i) for i,w in enumerate(old) for c in norm(w['text'])];bb=[(c,i) for i,w in enumerate(new) for c in norm(w['text'])];pairs={}
 for a,b,n in difflib.SequenceMatcher(None,[c for c,i in aa],[c for c,i in bb],autojunk=False).get_matching_blocks():
  for k in range(n):pairs.setdefault(aa[a+k][1],[]).append(bb[b+k][1])
 out={}
 for i,js in pairs.items():
  # Reject matching a short anchor to characters inside an unrelated word.
  unique=sorted(set(js))
  if len(js)==len(norm(old[i]['text'])) and norm(old[i]['text'])==''.join(norm(new[j]['text']) for j in unique) and unique==list(range(min(unique),max(unique)+1)):
   out[i]={'start':min(new[j]['start'] for j in unique),'end':max(new[j]['end'] for j in unique)}
 return out
rows={e['id']:e for e in read(B/'inputs.jsonl')};raw=D/'alignment_repairs_raw.jsonl'
if not raw.exists():raw.write_text((D/'alignment_repairs.jsonl').read_text())
results=[]
for z in read(raw):
 e=rows[z['id']];ws=e['tracks'][e['focus']]['words'];idx={w['id']:i for i,w in enumerate(ws)};li=idx[e['left_word']['id']];ri=idx[e['right_word']['id']]
 for p in z['probes']:
  count=p['neighbor_words'];words=ws[max(0,li-count):min(len(ws),ri+count+1)];a,b=p['window']
  words=[w for w in words if w['start']>=a and w['end']<=b];ids={w['id']:i for i,w in enumerate(words)};mp=mapping(words,p['words']);left=mp.get(ids.get(e['left_word']['id']));right=mp.get(ids.get(e['right_word']['id']))
  p.update(left=left,right=right,valid=bool(left and right and .015<=left['end']-left['start']<=2 and .015<=right['end']-right['start']<=2 and left['start']<right['start'] and abs(left['end']-e['anchor'])<1 and abs(right['start']-e['event_end'])<1),gap=round(right['start']-left['end'],3) if left and right else None)
 valid=[p for p in z['probes'] if p['valid']];clusters=[]
 for n in range(len(valid),1,-1):
  for c in itertools.combinations(valid,n):
   if len({p['neighbor_words'] for p in c})<2 or not any(p['transcript_mode']=='asr' for p in c):continue
   ll=[p['left']['end'] for p in c];rr=[p['right']['start'] for p in c]
   if max(ll)-min(ll)>.12 or max(rr)-min(rr)>.12 or len({p['gap']>.5 for p in c})!=1:continue
   l=statistics.median(ll);r=statistics.median(rr)
   if any(abs(p['left']['end']-l)>.2 or abs(p['right']['start']-r)>.2 or (p['gap']>.5)!=(r-l>.5) for p in valid):continue
   clusters.append(c)
  if clusters:break
 z.pop('accepted_boundary',None);z['repair_status']='still_unstable';z['validator']='exact_lexical_anchor_no_contradictory_valid_probe_v2'
 if clusters:
  c=clusters[0];l=statistics.median(p['left']['end'] for p in c);r=statistics.median(p['right']['start'] for p in c)
  z['accepted_boundary']={'left_end':l,'right_start':r,'gap':round(r-l,3),'votes':len(c),'windows':2,'includes_asr_transcript':True};z['repair_status']='stable_reestimated'
 results.append(z)
(D/'alignment_repairs.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in results))
from collections import Counter
print(Counter(z['repair_status'] for z in results));print('no_longer_eligible',sum(z.get('accepted_boundary',{}).get('gap',99)<=.5 for z in results))
