"""Measure actual served-model tokens; fallback only at complete-block boundaries."""
import copy,json,urllib.request

def fit_gap_payload(payload,prompt,endpoint,model,reserve=1024):
    value=copy.deepcopy(payload)
    token_url=endpoint.split('/v1/')[0]+'/tokenize'
    audit=[]
    for attempt in range(2):
        body=dict(model=model,messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(value))],add_generation_prompt=True)
        req=urllib.request.Request(token_url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as stream: measured=json.load(stream)
        count,maximum=int(measured['count']),int(measured['max_model_len'])
        audit.append(dict(input_tokens=count,max_model_len=maximum,reserved_output=reserve,margin=128))
        if count+reserve+128<=maximum:
            return value,dict(checks=audit,whole_block_fallback=bool(attempt),context_policy=value['context_policy'])
        if attempt:break
        blocks=value['context'];lo,hi=value['target']['absolute_gap']
        hits=[i for i,b in enumerate(blocks) if b['start']<=hi and b['end']>=lo]
        if not hits:raise ValueError('context_target_missing')
        first=min(hits);count=0
        while first>0 and count<6:
            first-=1;count+=len(blocks[first]['text'].split())>3
        kept=blocks[first:]
        if first==0:break
        value['context']=kept
        ids={b['block_id'] for b in kept}
        for field in ('other_near_target','other_before_right'):
            value[field]=[b for b in value.get(field,[]) if b['block_id'] in ids]
        value['context_policy'].update(mode='previous_3_exchanges_token_budget',history_truncated=True,
                                       first_block=kept[0]['block_id'],blocks=len(kept))
    raise ValueError('complete_context_exceeds_model_budget_no_sentence_truncation:'+json.dumps(audit))
