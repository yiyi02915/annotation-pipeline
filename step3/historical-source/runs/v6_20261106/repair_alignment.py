"""Collect audio-derived evidence; preserve original transcript until checks pass."""
import sys,json,gc,time,re
from pathlib import Path
sys.path.insert(0,'/data/L202500293/workspace/git/Qwen3-ASR')
import torch,soundfile as sf
from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
root=Path(__file__).resolve().parent
base=Path('/data/L202500493/duplex_postprocess/runtime/glm')
sys.path.insert(0,str(base))
from pipeline_core import alignment_flags
records=json.loads((root/'records.baseline.json').read_text())
targets=[];audio_cache={}
for r in records:
    for u in r['utterances']:
        if not alignment_flags(u) and not u.get('transcript_status'):continue
        path=Path('/data/L202500293/wzk/stage1/seamless_interaction_5dialogs')/r['conversation_id']/(r['conversation_id']+'_'+u['speaker']+'.wav')
        if str(path) not in audio_cache:
            data,sr=sf.read(path,dtype='float32');data=data.mean(axis=1) if data.ndim>1 else data
            audio_cache[str(path)]=(data,sr)
        data,sr=audio_cache[str(path)]
        peers=sorted([x for x in r['utterances'] if x['speaker']==u['speaker']],key=lambda x:x['start'])
        i=next(i for i,x in enumerate(peers) if x['utterance_id']==u['utterance_id'])
        windows=[]
        for pad in [0.7,2.0]:
            a=max(0,u['start']-pad);b=min(len(data)/sr,u['end']+pad)
            windows.append([round(a,3),round(b,3)])
        targets.append(dict(conversation_id=r['conversation_id'],utterance=u,audio=str(path),windows=windows,
            previous=peers[i-1] if i else None,next=peers[i+1] if i+1<len(peers) else None))
def save():
    (root/'alignment_evidence.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2),encoding='utf8')
asr=Qwen3ASRModel.from_pretrained('/data/L202500293/obed/models/qwen3-asr/checkpoints/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_new_tokens=256)
for t in targets:
    data,sr=audio_cache[t['audio']];t['asr']=[]
    for a,b in t['windows']:
        result=asr.transcribe(audio=(data[int(a*sr):int(b*sr)],sr),language='English')
        t['asr'].append(dict(window=[a,b],text=result[0].text))
        print(t['utterance']['utterance_id'],'ASR',a,b,result[0].text,flush=True);save()
del asr;gc.collect();torch.cuda.empty_cache()
aligner=Qwen3ForcedAligner.from_pretrained('/data/L202500293/obed/pipline_pre/qwenasr-code/checkpoint/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
for t in targets:
    data,sr=audio_cache[t['audio']];t['alignments']=[]
    texts=list(dict.fromkeys([t['utterance']['text']]+[x['text'] for x in t['asr'] if x['text'].strip()]))
    for text in texts:
        for a,b in t['windows']:
            try:
                aligned=aligner.align(audio=(data[int(a*sr):int(b*sr)],sr),text=text,language='English')
                words=[dict(word=w.text,start=round(a+w.start_time,3),end=round(a+w.end_time,3)) for w in aligned[0]]
                t['alignments'].append(dict(window=[a,b],text=text,words=words))
            except Exception as exc:t['alignments'].append(dict(window=[a,b],text=text,error=repr(exc)))
            save()
    print(t['utterance']['utterance_id'],'FA_COMPLETE',flush=True)
print('AUDIO_EVIDENCE_COMPLETE',flush=True)
