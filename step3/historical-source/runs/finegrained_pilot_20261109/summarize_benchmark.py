"""Compare two prediction sets without presenting agreement as accuracy."""
import collections,hashlib,json,statistics
from pathlib import Path
from benchmark_common import ROOT,LABELS,read_inputs
def predictions(name):
    p=ROOT/(name+'_predictions.jsonl')
    return {r['id']:r for r in map(json.loads,p.read_text().splitlines())} if p.exists() else {}
def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None
def main():
    rows=read_inputs();models={m:predictions(m) for m in ['qwen','jev']};out=[];tasks={}
    for task in LABELS:
        subset=[e for e in rows if e['task']==task];stats={}
        for name,preds in models.items():
            valid=[preds[e['id']] for e in subset if e['id'] in preds and preds[e['id']]['status']=='ok']
            stats[name]={'completed':len(valid),'labels':dict(collections.Counter(r['label'] for r in valid)),
                         'target_labels':sum(r['label'] in LABELS[task][:2] for r in valid)}
            if name=='jev':
                stats[name]['confidence_below_0_6']=sum(r['confidence']<.6 for r in valid)
                stats[name]['margin_below_0_15']=sum(r['margin']<.15 for r in valid)
                stats[name]['confidence_calibrated']=False
                if task=='overlap':
                    stats[name]['backchannel_with_barge_in_yes']=sum(r['label']=='BACKCHANNEL' and r['model_answers']['barge_in']['answer']=='yes' for r in valid)
        paired=[];matrix=collections.Counter();target_pairs=[]
        for e in subset:
            q,j=(models[m].get(e['id'],{}) for m in ['qwen','jev'])
            both=q.get('status')==j.get('status')=='ok'
            same=both and q['label']==j['label']
            if both:
                paired.append(same);matrix[q['label']+' | '+j['label']]+=1
                if q['label'] in LABELS[task][:2] and j['label'] in LABELS[task][:2]:target_pairs.append(same)
            out.append({'id':e['id'],'task':task,'conversation_id':e['conversation_id'],'focus':e['focus'],
                'anchor':e['anchor'],'event_end':e['event_end'],'qwen_label':q.get('label'),'jev_label':j.get('label'),
                'jev_confidence':j.get('confidence'),'jev_margin':j.get('margin'),'agreement':same if both else None,
                'consensus_label':q['label'] if same else 'UNCERTAIN','consensus_status':'model_agreement' if same else 'automatic_unresolved',
                'gold_label':None,'human_review_used':False,'audio_clip':e['audio_clip']})
        stats.update(paired_count=len(paired),agreement_count=sum(paired),
            agreement_rate=sum(paired)/len(paired) if paired else None,confusion_qwen_rows_jev_columns=dict(matrix),
            both_target_count=len(target_pairs),both_target_agreement=sum(target_pairs)/len(target_pairs) if target_pairs else None)
        tasks[task]=stats
    runtimes={m:read(m+'_runtime.json') for m in models}
    summary={'selected_per_task':1500,'total_candidates':3000,'tasks':tasks,'runtimes':runtimes,
        'complete':all(len(models[m])==3000 and all(x['status']=='ok' for x in models[m].values()) for m in models),
        'accuracy':None,'accuracy_evaluated':False,'reason':'No independent behavioral gold labels. Agreement and model confidence are not accuracy.',
        'source':read('audio_summary.json'),'input_audit':read('input_audit.json'),
        'limitations':['Supplied word alignment and speaker tracks, not independently re-aligned or verified VAD.',
                       'Offline context includes speech after the event; does not measure online latency or live policy quality.',
                       'Text-only Qwen vs audio-plus-text JEV; comparison is of these two configured pipelines.',
                       'Training split sampling with duration/quality filters is not representative class prevalence.',
                       'Model agreement alone does not establish training-label correctness.']}
    (ROOT/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'comparison_events.jsonl').open('w') as f:
        for r in out:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:summary[k] for k in ['complete','total_candidates','tasks','accuracy_evaluated']},indent=2))
if __name__=='__main__':main()
