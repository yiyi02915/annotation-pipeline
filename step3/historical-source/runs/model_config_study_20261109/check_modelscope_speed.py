import requests,time,json
url='https://modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para/resolve/master/model-v15-00001-of-00016.safetensors'
t=time.time()
with requests.get(url,headers={'Range':'bytes=0-33554431'},stream=True,timeout=20) as r:
 n=0
 for b in r.iter_content(1048576):
  n+=len(b)
  if n>=33554432:break
 print(r.status_code,n,time.time()-t,r.headers.get('Content-Range'))
url='https://modelscope.cn/api/v1/models/AdventistsAI/DuplexJev-32B-Para/repo/files?Revision=master&Recursive=true'
r=requests.get(url,timeout=20);obj=r.json();open('/data/L202500493/duplex_postprocess/runs/model_config_study_20261109/modelscope_manifest.json','w').write(json.dumps(obj));print('manifest',r.status_code,str(obj)[:1200])
