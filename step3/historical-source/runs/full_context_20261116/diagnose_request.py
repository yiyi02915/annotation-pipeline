import json,sys,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'step3'))
from gap_judge import PROMPT,COMPLETE_CONTEXT_RULES,PROPERTIES
x=json.loads((ROOT/'inputs_complete.json').read_text())[0]['gaps'][0]
body=dict(model='qwen38-27b',messages=[dict(role='system',content=PROMPT+COMPLETE_CONTEXT_RULES),dict(role='user',content=json.dumps(x))],max_tokens=1024,temperature=0,chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_schema','json_schema':{'name':'boundary','strict':True,'schema':{'type':'object','properties':PROPERTIES,'required':list(PROPERTIES),'additionalProperties':False}}})
try:
    req=urllib.request.Request('http://127.0.0.1:18771/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as f:print(json.load(f))
except urllib.error.HTTPError as e:print(e.read().decode())
