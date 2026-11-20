import json
from pathlib import Path
from transformers import AutoTokenizer
from run import D,RULES,events
tokenizer=AutoTokenizer.from_pretrained('/data2/202610/Qwen3.8-27B',local_files_only=True)
counts=[]
for e in events():
 messages=[{'role':'system','content':RULES},{'role':'user','content':json.dumps(e['payload'])}]
 # Reserve 256 tokens for the actual extra instructions, beyond the shared rules.
 rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
 count=len(tokenizer.encode(rendered,add_special_tokens=False))+256
 counts.append({'id':e['id'],'prompt_tokens_upper_estimate':count})
report={'count':len(counts),'max_prompt_tokens_upper_estimate':max(x['prompt_tokens_upper_estimate'] for x in counts),'all_fit_primary_1536':all(x['prompt_tokens_upper_estimate']+1536<=8192 for x in counts),'all_fit_arbiter_3072':all(x['prompt_tokens_upper_estimate']+3072<=8192 for x in counts),'largest':sorted(counts,key=lambda x:x['prompt_tokens_upper_estimate'],reverse=True)[:5]}
(D/'context_budget_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
