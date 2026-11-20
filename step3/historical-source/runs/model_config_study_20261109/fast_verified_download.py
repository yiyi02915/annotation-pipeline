"""Resume the verified identical mirror into one file per shard; validate every SHA256."""
import concurrent.futures,hashlib,json,os,signal,time
from pathlib import Path
import requests
D=Path(__file__).resolve().parent;M=D.parents[1]/'models/DuplexJev-32B-Para'
files=json.loads((D/'verified_mirror_files.json').read_text());assert len(files)==16
pid=1450104
try:
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 assert b'download_candidates.py' in cmd
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(.1)
except FileNotFoundError:pass
plans=[];jobs=[]
for meta in files:
 final=M/meta['name'];part=M/(meta['name']+'.fastpartial');state=M/(meta['name']+'.ranges.json')
 if final.exists():plans.append((meta,final,None));continue
 if state.exists():s=json.loads(state.read_text())
 else:
  partials=list((M/'.cache/huggingface/download').glob('*.'+meta['sha256']+'.incomplete'))
  assert len(partials)<=1
  prefix=partials[0].stat().st_size if partials else 0
  assert prefix<=meta['size']
  if partials:partials[0].rename(part)
  else:part.touch()
  s={'prefix':prefix,'done':[]};state.write_text(json.dumps(s))
 url='https://modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para/resolve/'+meta['ms_revision']+'/'+meta['name']
 # Resolve once; subsequent byte-range requests use the returned object URL.
 response=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=30)
 response.raise_for_status();resolved=response.url;response.close()
 fd=os.open(part,os.O_RDWR);os.ftruncate(fd,meta['size'])
 plan=dict(meta=meta,part=part,state=state,s=s,fd=fd,url=resolved);plans.append((meta,part,plan))
 for start in range(s['prefix'],meta['size'],32*1024*1024):
  if start not in s['done']:jobs.append((plan,start,min(start+32*1024*1024,meta['size'])))
jobs.sort(key=lambda item: ((item[1]-item[0]['s']['prefix'])//(32*1024*1024),item[0]['meta']['name']))
print('RESUME_BYTES',sum(p[2]['s']['prefix'] if p[2] else p[0]['size'] for p in plans),'RANGES',len(jobs),flush=True)
def fetch(job):
 p,start,end=job
 for attempt in range(5):
  try:
   with requests.get(p['url'],headers={'Range':f'bytes={start}-{end-1}'},stream=True,timeout=(20,60)) as response:
    response.raise_for_status();assert response.status_code==206
    assert response.headers.get('Content-Range','').startswith(f'bytes {start}-{end-1}/')
    offset=start
    for data in response.iter_content(1024*1024):
     assert offset+len(data)<=end
     view=memoryview(data)
     while view:
      written=os.pwrite(p['fd'],view,offset);offset+=written;view=view[written:]
    assert offset==end
   return p,start,end-start
  except Exception:
   if attempt==4:raise
   time.sleep(1+attempt)
t=time.time();downloaded=0
with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
 for i,(p,start,n) in enumerate(pool.map(fetch,jobs)):
  p['s']['done'].append(start);p['state'].write_text(json.dumps(p['s']));downloaded+=n
  if (i+1)%32==0:print('RANGES_DONE',i+1,'BYTES',downloaded,'SECONDS',round(time.time()-t),flush=True)
checks=[]
for meta,path,p in plans:
 if p:os.fsync(p['fd']);os.close(p['fd'])
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 assert path.stat().st_size==meta['size'] and h.hexdigest()==meta['sha256'],meta['name']
 if p:path.rename(M/meta['name'])
 checks.append(dict(name=meta['name'],sha256=h.hexdigest(),size=meta['size']));print('VERIFIED',meta['name'],flush=True)
(D/'weights_sha256_verified.json').write_text(json.dumps(checks,indent=2))
with (D/'download.log').open('a') as f:f.write('\nDOWNLOAD_COMPLETE DuplexJev-32B-Para\n')
print('VERIFIED_DOWNLOAD_COMPLETE',time.time()-t,flush=True)
