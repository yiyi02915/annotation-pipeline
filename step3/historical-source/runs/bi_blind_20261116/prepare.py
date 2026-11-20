import hashlib,json,random,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
DATA=Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd')
sys.path.insert(0,str(ROOT/'step3'))
from pipeline import WordTimeline,windows
from dialogue_context import DialogueContext
from event_extractor import extract_candidates
from audio_judge import audio_duration,render_mix
def save(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 lock=json.loads((ROOT/'source_lock.json').read_text())
 assert all(hashlib.sha256((ROOT/'step3'/n).read_bytes()).hexdigest()==h for n,h in lock['source_hashes'].items())
 excluded=set();audit=[];errors=[];pat=re.compile(rb'V\d+_S\d+_I\d+')
 for f in sorted((BASE/'runs').rglob('*')):
  if ROOT in f.parents or not f.is_file() or f.is_symlink():continue
  if any(t in f.parts for t in ('site-packages','.git','node_modules','chrome-profile','__pycache__')):continue
  if f.suffix not in ('.json','.jsonl','.txt','.csv','.md','.py','.log'):continue
  hits=set()
  try:
   with f.open('rb') as stream:
    tail=b''
    while True:
     chunk=stream.read(2**20)
     if not chunk:break
     buf=tail+chunk;hits.update(x.decode() for x in pat.findall(buf));tail=buf[-100:]
  except OSError as e:errors.append(dict(path=str(f),error=str(e)));continue
  if hits:excluded.update(hits);audit.append(dict(path=str(f),ids=len(hits)))
 sessions={c.split('_I')[0] for c in excluded}
 save('exclusion_audit.json',dict(ids=sorted(excluded),sessions=sorted(sessions),files=audit,errors=errors))
 print('EXCLUSIONS',len(excluded),len(sessions),len(audit),'errors',len(errors),flush=True)
 indices=[]
 with (DATA/'training/index_v1/train.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   if row['interaction_id'].split('_I')[0] not in sessions:indices.append(row)
 rng=random.Random(2026111602);rng.shuffle(indices)
 save('sampling_plan.json',dict(seed=2026111602,target=30,strata={'question_end':10,'repeated_receipts':10,'ongoing_entry':10},
   unused_index_rows=len(indices),exclude_same_session=True,reference_author='Codex semantic adjudication before predictions',
   scope='challenge-stratified sample, not unbiased population prevalence',frozen_source=lock['source_hashes']))
 print('ELIGIBLE_INDEX',len(indices),flush=True)
 counts=dict(question_end=0,repeated_receipts=0,ongoing_entry=0);out=[];seen=set();rejects=[]
 receipts={'yes','yeah','right','okay','ok','oh','uh-huh','mm-hmm','exactly','mhm'}
 with (DATA/'seamless_fd_data/metadata.jsonl').open('rb') as f:
  for row in indices:
   cid=row['interaction_id'];session=cid.split('_I')[0]
   if session in seen or len(row['participants'])!=2:continue
   paths={p['sample_id'].rsplit('_',1)[1]:Path(p['audio_path'].replace('/hbt_share02/202610/','/data2/202610/')) for p in row['participants']}
   if not all(p.is_file() for p in paths.values()):continue
   f.seek(row['metadata_byte_offset']);meta=json.loads(f.read(row['metadata_bytes']))
   if meta['interaction_id']!=cid or meta['participant_count']!=2:continue
   duration=min(p['duration_seconds'] for p in row['participants']);full={}
   for p in meta['participants']:
    sp=p['sample_id'].rsplit('_',1)[1]
    full[sp]=sorted([dict(id=f'{sp}:w{i}',text=w['word'],start=float(w['start_sec']),end=float(w['end_sec'])) for i,w in enumerate(p['words'])
       if 0<=float(w['start_sec'])<float(w['end_sec'])<=duration+.01 and w['word'].strip()],key=lambda w:(w['start'],w['end']))
   if set(full)!=set(paths) or not all(full.values()):continue
   proposals=[]
   for sp,words in full.items():
    other=full[next(k for k in full if k!=sp)]
    for i,w in enumerate(words):
     a=w['start']
     if not 8<a<duration-12:continue
     overlap=[v for v in other if v['start']<a<v['end']]
     if not overlap:continue
     if counts['question_end']<10 and any(v['text'].rstrip().endswith('?') and v['end']-a<=.35 for v in overlap):
      proposals.append(('question_end',a,w['id']))
     if counts['repeated_receipts']<10 and w['text'].lower().strip('.,!?') in receipts and any(v['text'].lower().strip('.,!?') in receipts and v['start']-a<2 for v in words[i+1:i+5]):
      proposals.append(('repeated_receipts',a,w['id']))
     if counts['ongoing_entry']<10 and i and a-words[i-1]['end']>.4:
      following=[v for v in words[i:i+20] if v['start']<a+4]
      if len(following)>=8 and any(v['text'].rstrip().endswith(('.', '?', '!')) for v in following[:-2]) and any(a+.4<v['start']<a+3 for v in other):
       proposals.append(('ongoing_entry',a,w['id']))
   rng.shuffle(proposals)
   for stratum,a,wid in proposals:
    core=[round(a-.8,6),round(a+4.2,6)];interval=[round(a-5,6),round(a+9,6)]
    cs=list(windows(cid,WordTimeline(full),duration,core=core,analysis=interval))
    if not cs:continue
    case=cs[0][0];cand=extract_candidates(case,refined=True,floor_checks=True)
    targets=[c for c in cand if core[0]<=c['start']<core[1]]
    if not 2<=len(targets)<=12 or any(c['alignment_flags'] for c in targets):continue
    if any(w['end']-w['start']>2 for ws in case['streams'].values() for w in ws):continue
    tracks={k:(p,0) for k,p in paths.items()}
    try:audio_duration(tracks)
    except ValueError as e:rejects.append(dict(id=cid,reason=str(e)));continue
    case=DialogueContext(full).bi(case);case['audio_text_scope']='local'
    number=f'B{len(out)+1:02}';case['number']=number
    (ROOT/'audio').mkdir(exist_ok=True)
    render_mix(tracks,interval,ROOT/'audio'/f'{number}.wav')
    out.append(dict(number=number,stratum=stratum,conversation_id=cid,case=case,candidates=cand,
       tracks={k:[str(p),ch] for k,(p,ch) in tracks.items()},anchor_word_id=wid))
    counts[stratum]+=1;seen.add(session);print('SELECTED',number,stratum,cid,len(targets),flush=True);break
   if all(v==10 for v in counts.values()):break
 assert all(v==10 for v in counts.values()),counts
 save('inputs.json',out);save('sampling_summary.json',dict(counts=counts,records=len(out),rejections=rejects,excluded_dialogues=len(excluded),excluded_sessions=len(sessions)))
 dossier=[]
 for r in out:
  c=r['case'];lines=[f"=== {r['number']} {r['conversation_id']} {r['stratum']} core={c['target']} ==="]
  for b in c['dialogue_context']:lines.append(f"[{b['start']:.3f}-{b['end']:.3f}] {b['speaker']}: {b['text']}")
  lines.append('CANDIDATES (asterisk = scored core)')
  for i,x in enumerate(r['candidates']):
   core=c['target'][0]<=x['start']<c['target'][1];lines.append(f"{'*' if core else ' '}C{i+1} {x['initiator']}->{x['recipient']} [{x['start']:.3f}-{x['end']:.3f}] {x['text']}")
  dossier.append('\n'.join(lines))
 (ROOT/'reference_dossier.txt').write_text('\n\n'.join(dossier))
 print('PREPARE_COMPLETE',counts,flush=True)
if __name__=='__main__':main()
