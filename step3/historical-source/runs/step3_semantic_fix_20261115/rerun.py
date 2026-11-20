"""New prompt/feature comparison. Overlap audio votes are frozen across variants."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

out=Path(__file__).resolve().parent
sys.path.insert(0,str(out/'step3'))
import pipeline
from audio_judge import audio_duration
from input_seamless import load_published

parser=argparse.ArgumentParser()
parser.add_argument('--audio',action='store_true')
parser.add_argument('--name',default='run_text')
options=parser.parse_args()
folder=out/options.name;folder.mkdir();(folder/'recordings').mkdir()
base=out.parent/'step3_validation_20261115/run/recordings'
votes={e['candidate_id']:e['mix'] for p in base.glob('*.json')
       for w in json.loads(p.read_text())['windows'] for e in w['audio_votes']}
audio=json.loads((out/'audio_votes.json').read_text()) if options.audio else None
if audio:
    assert len(audio['gap'])==61 and 'inference_wall_seconds' in audio
call=pipeline.request
def request(payload,prompt,**kwargs):
    if audio:
        payload=dict(payload,audio_boundary_evidence=audio['gap'][payload['id']]['answer'])
    return call(payload,prompt,**kwargs)
pipeline.request=request
if audio:
    pipeline.windows=lambda *args,**kwargs: iter(())
class CachedJudge:
    def vote(self,case,candidate,waveform,lang):
        return copy.deepcopy(votes[candidate['candidate_id']])
judge=CachedJudge()
rows=[json.loads(line) for line in (out/'pairs.jsonl').read_text().splitlines()]
args=SimpleNamespace(output=folder,qwen_endpoint='http://127.0.0.1:18770/v1/chat/completions',
                     qwen_model='qwen38-27b',review_fraction=.2,language='en')
inputs=list(load_published(Path('/data2/202610/tffd/datasets/full_duplex_speech/seamless_fd'),rows))
results=[];start=time.monotonic()
def process(index,values,pool):
    row,streams,tracks,problem=values
    assert not problem,problem
    duration=audio_duration(tracks)
    for words in streams.values():words[:]=[w for w in words if w['end']<=duration+.001]
    result=pipeline.annotate(row,streams,tracks,duration,judge,args,pool)
    if audio:
        frozen=json.loads((out/'run_text/recordings'/f'{index:08d}.json').read_text())
        for key in ('events','windows','overlap_candidates'):
            result[key]=frozen[key]
        result['failures'].extend(f for f in frozen['failures'] if f['stage'].startswith('overlap'))
        result['status']='partial' if result['failures'] else 'complete'
    pipeline.write_json(folder/'recordings'/f'{index:08d}.json',result)
    print(index,result['status'],flush=True)
    return result
with ThreadPoolExecutor(4) as gaps,ThreadPoolExecutor(2) as recordings:
    for future in as_completed([recordings.submit(process,i,item,gaps) for i,item in enumerate(inputs,1)]):
        results.append(future.result())
summary=pipeline.summarize(results,len(rows),time.monotonic()-start)
summary['cached_overlap_votes']=summary['model_requests']['jev']
summary['model_requests']['jev']=0
if audio:
    summary['cached_overlap_qwen_requests']=summary['model_requests']['overlap_qwen']
    summary['model_requests']['overlap_qwen']=0
summary['first_labels_per_minute']=None
summary['first_wall_seconds']=None
summary['audio_gap_precomputation']=dict(requests=len(audio['gap']),seconds=audio['inference_wall_seconds'],load_seconds=audio['load_seconds']) if audio else None
summary['experiment_note']='Overlap JEV votes reused identically; Qwen rerun live. Two recordings concurrent; first-only throughput not measured. Gap audio precomputation excluded from run wall time.'
pipeline.write_json(folder/'summary.json',summary)
print('RUN_COMPLETE',flush=True)
