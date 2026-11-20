import json,urllib.request
url='http://127.0.0.1:18771/tokenize'
body=dict(model='qwen38-27b',messages=[dict(role='user',content='hello')],add_generation_prompt=True)
req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req) as f:r=json.load(f)
print({k:v for k,v in r.items() if k not in ['tokens','token_strs']})
