import requests,time
url='https://modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para/resolve/master/model-v15-00001-of-00016.safetensors'
t=time.time()
try:
 with requests.get(url,headers={'Range':'bytes=0-1048575'},stream=True,timeout=15) as r:
  data=r.raw.read(1048576);print(r.status_code,len(data),time.time()-t,r.headers.get('Content-Range'))
except Exception as ex: print(type(ex).__name__,str(ex)[:250])
