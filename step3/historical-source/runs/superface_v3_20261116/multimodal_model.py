"""Differentiable JEV audio+text path with frozen audio modules and task LoRA."""
import json
import torch
from duplexjev import Decider
from peft import LoraConfig, get_peft_model, PeftModel

CODES={'PAUSE':'P','TURN_TRANSITION':'T','INTERRUPT':'I','BACKCHANNEL':'B','NEITHER':'N'}
OPTIONS={'gap':['PAUSE','TURN_TRANSITION'],'overlap':['INTERRUPT','BACKCHANNEL','NEITHER']}
from audit_multimodal import RULES, evidence


class Model:
    def __init__(self):
        self.engine=Decider.from_pretrained('/data/L202500493/duplex_postprocess/models/DuplexJev-4B-Para',device='cuda:0',attn_implementation='sdpa')
        self.model=self.engine.model
        self.model.requires_grad_(False)
        self.model.eval()
        self.audio_forward_calls=0
        self.original_prepare=self.model._prepare_audio_embeds
        def counted(**kwargs):
            self.audio_forward_calls+=1
            return self.original_prepare(**kwargs)
        self.model._prepare_audio_embeds=counted

    def adapter(self, checkpoint=None):
        if hasattr(self.model.language_model,'peft_config'):
            self.model.language_model=self.model.language_model.unload()
            # PEFT unload removes LoRA modules but may leave the metadata attribute.
            # Clear only that stale metadata before installing the next task adapter.
            if hasattr(self.model.language_model,'peft_config'):
                delattr(self.model.language_model,'peft_config')
        self.model.requires_grad_(False)
        if checkpoint:
            self.model.language_model=PeftModel.from_pretrained(self.model.language_model,str(checkpoint),is_trainable=True)
        else:
            self.model.language_model=get_peft_model(self.model.language_model,LoraConfig(r=8,lora_alpha=32,lora_dropout=0,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj']))
        self.model.language_model.config.use_cache=False

    def encode(self,row,audio=True):
        ctx=RULES+'\n'+json.dumps(evidence(row),ensure_ascii=False)
        question='\nClassify only the target. Return exactly one code: '+', '.join(CODES[x]+'='+x for x in OPTIONS[row['task']])+'.\n'
        if audio:
            encoded=self.engine._encode_prefix({'context':ctx,'audio':row['audio']},'en')
            suffix=self.engine.tok(question+encoded['tail'],add_special_tokens=False)['input_ids']
            batch=self.engine._collate([(encoded['ids']+suffix,encoded['audio'])])
            assert batch['audio_token_len'].sum()>0
        else:
            head,tail=self.engine._head_tail(ctx,'','en')
            ids=self.engine.tok(head+question+tail,add_special_tokens=False)['input_ids']
            batch=dict(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device='cuda'))
        if batch['input_ids'].shape[1]>6500: raise ValueError('oversized_target_context')
        return batch

    def logits(self,row,audio=True):
        batch=self.encode(row,audio)
        # Full multimodal forward: audio encoder/projector are frozen, not removed.
        return self.model(**batch,use_cache=False).logits[0,-1].float()

    def loss(self,row,audio=True):
        ids=self.engine.tok(CODES[row['label']],add_special_tokens=False)['input_ids']
        assert len(ids)==1
        return torch.nn.functional.cross_entropy(self.logits(row,audio).unsqueeze(0),torch.tensor(ids,device='cuda'))

    def scores(self,row,audio=True):
        logits=self.logits(row,audio)
        ids=[self.engine.tok(CODES[x],add_special_tokens=False)['input_ids'][0] for x in OPTIONS[row['task']]]
        return logits[ids]

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def assert_frozen(self):
        assert all(not p.requires_grad and p.grad is None for n,p in self.model.named_parameters() if not n.startswith('language_model.'))
        assert all('lora_' in n for n,p in self.model.named_parameters() if p.requires_grad)
