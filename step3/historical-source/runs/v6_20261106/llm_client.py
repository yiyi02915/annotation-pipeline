import json,hashlib,time,threading
from pathlib import Path
import requests

def obj_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
def enum(values):return dict(type='string',enum=values)
BOUNDARY_SCHEMA=obj_schema(dict(code=enum(list('ABCDEF')),left_action={'type':'string'},right_action={'type':'string'},reason={'type':'string'}))
EVENT_SCHEMA=obj_schema(dict(interaction_type=enum(['BACKCHANNEL','INTERRUPT','TURN','OTHER','UNRESOLVED']),function=enum(['ACKNOWLEDGE','STANCE','REPAIR','REDIRECT','OTHER','UNKNOWN']),floor_completion=enum(['COMPLETE','INCOMPLETE','UNKNOWN']),floor_response=enum(['CONTINUE','YIELD','UNKNOWN']),content_uptake=enum(['ADDRESSED','NOT_ADDRESSED','UNKNOWN']),response_context_index={'type':'integer'},floor_quote={'type':'string'},incoming_quote={'type':'string'},reason={'type':'string'}))
EVENT_SCHEMA['properties']['secondary_function']=enum(['NONE','ACKNOWLEDGE','STANCE','REPAIR','REDIRECT','OTHER'])
EVENT_SCHEMA['required'].append('secondary_function')

class Client:
    def __init__(self,root,url='http://127.0.0.1:18351/v1',model='glm-judge'):
        self.root=Path(root);self.url=url;self.model=model;self.lock=threading.Lock();self.cache={}
        self.log=self.root/'model_calls.jsonl'
        if self.log.exists():
            for line in self.log.read_text(encoding='utf8').splitlines():
                try:r=json.loads(line);self.cache[r['request_hash']]=r
                except (ValueError,KeyError):pass
    def call(self,system,payload,schema,tag):
        body=dict(model=self.model,messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],temperature=0,max_tokens=2048,chat_template_kwargs=dict(reasoning_effort='low',clear_thinking=True),response_format=dict(type='json_schema',json_schema=dict(name='annotation',strict=True,schema=schema)))
        if self.model.startswith('qwen'):
            body['chat_template_kwargs']={'enable_thinking':False};body['max_tokens']=640
        initial_body=json.loads(json.dumps(body))
        key=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if key in self.cache and self.cache[key].get('parsed') is not None:return self.cache[key]
        errors=[];start=time.time();r=None;parsed=None
        for attempt in range(2):
            try:
                response=requests.post(self.url+'/chat/completions',json=body,timeout=600)
                response.raise_for_status();r=response.json();choice=r['choices'][0]
                if choice.get('finish_reason')=='length':
                    errors.append('truncated');body['max_tokens']=4096;continue
                content=choice['message'].get('content') or ''
                parsed=json.loads(content)
                if not isinstance(parsed,dict):raise ValueError('not object')
                break
            except Exception as exc:
                parsed=None
                detail=response.text[:400] if 'response' in locals() and not response.ok else str(exc)[:400]
                errors.append(type(exc).__name__+':'+detail)
                if attempt==0:time.sleep(1)
        record=dict(request_hash=key,request_hash_basis='initial_request_before_retry',initial_request=initial_body,actual_request_hash=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),tag=tag,model=self.model,parsed=parsed,raw=r,errors=errors,seconds=round(time.time()-start,3),request=body)
        with self.lock:
            with self.log.open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            self.cache[key]=record
        return record
