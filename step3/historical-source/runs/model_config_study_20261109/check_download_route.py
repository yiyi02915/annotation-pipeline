import urllib.request,time,json
urls=[('modelscope','https://modelscope.cn/models/AdventistsAI/DuplexJev-32B-Para/resolve/master/model-v15-00001-of-00016.safetensors'),('huggingface','https://huggingface.co/adventists-ai/DuplexJev-32B-Para/resolve/757f9df3bf72ffc5b7a708696d47982c39d48d8c/model-v15-00001-of-00016.safetensors')]
for name,url in urls:
 t=time.time()
 try:
  req=urllib.request.Request(url,headers={'Range':'bytes=0-1048575'})
  with urllib.request.urlopen(req,timeout=12) as res:
   data=res.read(1048576);print(name,res.status,len(data),time.time()-t,res.headers.get('Content-Range'),flush=True)
 except Exception as ex:print(name,type(ex).__name__,str(ex)[:100],flush=True)
