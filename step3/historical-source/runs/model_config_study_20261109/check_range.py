import requests,json,time
from pathlib import Path
f=json.loads(Path('/data/L202500493/duplex_postprocess/runs/model_config_study_20261109/verified_mirror_files.json').read_text())[0]
u='https://modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para/resolve/'+f['ms_revision']+'/'+f['name']
r=requests.get(u,headers={'Range':'bytes=0-0'},stream=True,timeout=15); url=r.url;print('resolve',r.status_code,r.headers.get('Content-Range'),flush=True);r.close()
for a,b in [(100000000,101048575),(200000000,201048575)]:
 t=time.time();r=requests.get(url,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=15);print(r.status_code,r.headers.get('Content-Range'),r.headers.get('Content-Length'),time.time()-t,flush=True);r.close()
