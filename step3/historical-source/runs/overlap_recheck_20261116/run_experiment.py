"""Fresh overlap-only timing with event-level, reference-blind automatic reviews."""
import copy,hashlib,json,os,sys,time,traceback,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
SOURCE=BASE/'runs/step3_semantic_fix_20261115'
sys.path.insert(0,str(ROOT/'step3'))
from audio_judge import MixedAudioJudge,render_mix
from event_extractor import extract_candidates,finalize
from run_multi import classify

LABELS=['BACKCHANNEL','INTERRUPT','NEITHER']
PROMPT='''Judge one directed intervention at its exact onset in a TWO-person conversation.
BACKCHANNEL: listener acknowledgment/stance supporting the ongoing speaker without claiming the floor.
INTERRUPT: substantive attempt to claim/change the floor before the other has completed or yielded. It need not succeed.
NEITHER: ordinary answer after completion, own-floor continuation, or a continuation of an already-started same-direction event rather than a new onset.
Check prior completion using semantic context, not punctuation or overlap alone. Follow the target speaker and precise onset, not a later overlap. Opposite-direction re-entry is a separate event. Local listener feedback inside an interruption can be a distinct backchannel. Brief feedback must not automatically reset floor ownership. Reject normal end-of-question answers as interruptions. Do not assume a third speaker from channel/ASR problems.
Return label, prior_complete (boolean), independent_entry (boolean), function, evidence (a verbatim phrase copied from supplied target/context), and concise reason. Audio is supporting evidence and can be wrong. Decide from the supplied context; no reference label is supplied.'''
SCHEMA={'type':'object','properties':{'label':{'type':'string','enum':LABELS},'prior_complete':{'type':'boolean'},'independent_entry':{'type':'boolean'},'function':{'type':'string','enum':['listener_feedback','floor_claim','ordinary_response','own_continuation']},'evidence':{'type':'string'},'reason':{'type':'string'}},'required':['label','prior_complete','independent_entry','function','evidence','reason'],'additionalProperties':False}

def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def record(event,**kw):
    with (ROOT/'checkpoints.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=now(),event=event,**kw),ensure_ascii=False)+'\n')
    print(event,json.dumps(kw),flush=True)
def brief_attempts(result):return [{k:v for k,v in a.items() if k!='response'} for a in result.get('attempts',[])]
def label_of(result,cid):
    return next((e['label'] for e in result.get('events',[]) if e['candidate_ids'][0]==cid),'NEITHER') if result.get('status')=='ok' else None
def review_payload(case,candidate,audio,short):
    a=max(case['interval'][0],candidate['start']-4) if short else case['interval'][0]
    b=min(case['interval'][1],max(candidate['end']+2,candidate['start']+4)) if short else case['interval'][1]
    ctx=[s for s in case.get('extended_context',case['context']) if s['end']>a and s['start']<b]
    target={k:candidate[k] for k in ['candidate_id','initiator','recipient','start','end','text','other_active_run_at_onset','overlap_seconds']}
    return dict(analysis_interval=[a,b],target=target,context=ctx,audio_evidence=audio)
def request_review(payload,mode):
    body=dict(model='qwen38-27b',messages=[dict(role='system',content=PROMPT+('\nUse the local window to resolve the exact target.' if mode=='short' else '\nIndependently verify the target against the full surrounding conversation.')),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],temperature=0,max_tokens=900,chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_schema','json_schema':{'name':'overlap_review','strict':True,'schema':SCHEMA}})
    attempts=[];start=time.monotonic()
    for _ in range(2):
        t=time.monotonic()
        try:
            req=urllib.request.Request('http://127.0.0.1:18770/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=240) as f:resp=json.load(f)
            choice=resp['choices'][0]
            if choice['finish_reason']!='stop':raise ValueError('unfinished_generation')
            answer=json.loads(choice['message']['content'])
            if answer.get('label') not in LABELS or not all(k in answer for k in SCHEMA['required']):raise ValueError('invalid_schema')
            evidence=answer['evidence'].strip().lower()
            available=[payload['target']['text'],*[s['text'] for s in payload['context']]]
            grounded=bool(evidence) and any(evidence in s.lower() for s in available)
            attempts.append(dict(seconds=time.monotonic()-t,status='ok',usage=resp.get('usage')))
            return dict(status='ok',answer=answer,evidence_grounded=grounded,seconds=time.monotonic()-start,attempts=attempts)
        except Exception as exc:attempts.append(dict(seconds=time.monotonic()-t,status='error',error=str(exc)))
    return dict(status='error',seconds=time.monotonic()-start,attempts=attempts)

def score(first,final,reviews,refs):
    details=[]
    for cid,refrow in refs['recordings'].items():
        for target in refrow.get('events',[]):
            if target['label'] not in LABELS:continue
            uid=target['candidate_id'];f=first.get(uid);z=final.get(uid,f);review=reviews.get(uid,{})
            details.append(dict(id=uid,conversation_id=cid,reference=target['label'],first=f,final=z,reviewed=bool(review),review_complete=review.get('review_complete',False),status=review.get('status','not_triggered')))
    n=len(details)
    metrics={}
    for phase in ['first','final']:
        correct=sum(d[phase]==d['reference'] for d in details)
        metrics[phase]=dict(total=n,valid=sum(d[phase] in LABELS for d in details),correct=correct,agreement=correct/n,errors=n-correct,classes={})
        for label in LABELS:
            group=[d for d in details if d['reference']==label];pred=[d for d in details if d[phase]==label];hit=sum(d[phase]==label for d in group)
            metrics[phase]['classes'][label]=dict(reference=len(group),correct=hit,predicted=len(pred),false_positive=len(pred)-hit)
    metrics.update(changed=sum(d['first']!=d['final'] for d in details),fixed=sum(d['first']!=d['reference'] and d['final']==d['reference'] for d in details),regressed=sum(d['first']==d['reference'] and d['final']!=d['reference'] for d in details),reviewed=sum(d['reviewed'] for d in details),review_complete=sum(d['review_complete'] for d in details))
    return dict(metrics=metrics,details=details)

def main():
    inputs=json.loads((SOURCE/'inputs.json').read_text());ref_path=SOURCE/'benchmark_reference.json'
    config=dict(started_at=now(),source=str(SOURCE),reference_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),qwen_parallel=2,audio_parallel=1,gpu='CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES',''),audio_cache=False,review_gate='alignment flags OR audio/text disagreement OR audio margin below .2; expected missing audio is a risk; select all triggered core candidates',review_accept='two grounded text views and fresh local audio all agree; preserve label on disagreement/failure/structural rejection',timing='loaded-model batch wall time, includes mixing and fresh JEV + Qwen; separate review phase; excludes FA/model loading',source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'step3').glob('*.py')},review_prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest())
    save(ROOT/'config_locked.json',config);record('CONFIG_LOCKED',recordings=len(inputs))
    t=time.monotonic();judge=MixedAudioJudge(BASE/'models/DuplexJev-32B-Para','cuda:0');load_seconds=time.monotonic()-t
    record('MODEL_READY',load_seconds=load_seconds)
    (ROOT/'audio').mkdir(exist_ok=True)
    first_results={};tasks=[];windows={};first_start=time.monotonic();first_utc=now()
    def text_first(case,votes,started):
        result=classify(case,audio=votes,concise=True,refined=True,floor_checks=True,recover_invalid_merge=True)
        result['attempts']=brief_attempts(result);result['audio_votes']=votes['events'];result['first_started_offset']=started-first_start;result['first_finished_offset']=time.monotonic()-first_start;result['first_latency_seconds']=time.monotonic()-started
        save(ROOT/'first'/f"{hashlib.sha256(case['id'].encode()).hexdigest()[:16]}.json",result)
        return case['id'],result
    with ThreadPoolExecutor(2) as pool:
        for inp in inputs:
            for case in inp['cases']:
                started=time.monotonic();tracks={k:(Path(v[0]),v[1]) for k,v in inp['tracks'].items()}
                key=hashlib.sha256(case['id'].encode()).hexdigest()[:16];mix=render_mix(tracks,case['interval'],ROOT/'audio'/f'{key}.wav')
                candidates=extract_candidates(case,refined=True,floor_checks=True)
                required=[c for c in candidates if c['other_active_run_at_onset'] and c['overlap_seconds']>0]
                votes=dict(clip_id=case['id'],events=[])
                for c in required:
                    t=time.monotonic();vote=judge.vote(case,c,mix,'en')
                    votes['events'].append(dict(candidate_id=c['candidate_id'],mix=vote,seconds=time.monotonic()-t))
                windows[case['id']]=dict(case=case,candidates=candidates,tracks=tracks,record=inp['row'],source_duration=inp['duration'])
                tasks.append(pool.submit(text_first,case,votes,started));record('AUDIO_WINDOW_DONE',window=case['id'],votes=len(votes['events']))
        for future in as_completed(tasks):
            key,result=future.result();first_results[key]=result;record('FIRST_WINDOW_DONE',window=key,status=result['status'])
    first_wall=time.monotonic()-first_start;record('FIRST_PHASE_DONE',windows=len(first_results),wall_seconds=first_wall)
    save(ROOT/'first_results.json',first_results)
    initial={};review_specs=[];all_candidates={};event_state=[]
    for key,w in windows.items():
        result=first_results[key];votes={v['candidate_id']:v['mix'] for v in result.get('audio_votes',[])}
        for c in w['candidates']:
            if not w['case']['target'][0]<=c['start']<w['case']['target'][1]:continue
            uid=c['candidate_id'];label=label_of(result,uid);initial[uid]=label;all_candidates[uid]=(key,c)
            flags=list(c.get('alignment_flags',[]));vote=votes.get(uid)
            if label is None:flags.append('first_failed')
            if vote:
                if vote['answer']!=label:flags.append('text_audio_disagreement')
                p=sorted(vote.get('probs',{}).values(),reverse=True)
                if len(p)>1 and p[0]-p[1]<.2:flags.append('small_audio_option_margin')
            elif c['other_active_run_at_onset'] and c['overlap_seconds']>0:flags.append('expected_audio_missing')
            row=dict(id=uid,conversation_id=w['case']['conversation_id'],window=key,start=c['start'],end=c['end'],initiator=c['initiator'],recipient=c['recipient'],first_label=label,risk_reasons=flags,review_selected=bool(flags),review_status='pending' if flags else 'not_triggered')
            event_state.append(row)
            if flags:review_specs.append((uid,key,c))
    save(ROOT/'event_status_first.json',event_state)
    save(ROOT/'review_selection_locked.json',dict(reference_blind=True,count=len(review_specs),candidate_count=len(initial),ids=[x[0] for x in review_specs],policy=config['review_gate']))
    reviews={};review_start=time.monotonic()
    def text_review(uid,key,c,vote,case,started,audio_seconds):
        short=request_review(review_payload(case,c,vote,True),'short')
        full=request_review(review_payload(windows[key]['case'],c,vote,False),'full')
        complete=short['status']==full['status']=='ok';labels=[x.get('answer',{}).get('label') for x in [short,full]]
        grounded=complete and short['evidence_grounded'] and full['evidence_grounded']
        proposal=labels[0] if grounded and labels[0]==labels[1]==vote.get('answer') else None
        row=dict(id=uid,window=key,first_label=initial[uid],short=short,full=full,audio=vote,audio_seconds=audio_seconds,review_complete=complete,proposal=proposal,status='consensus' if proposal else 'disagreement_or_ungrounded' if complete else 'failed',seconds=time.monotonic()-started,started_offset=started-review_start,finished_offset=time.monotonic()-review_start)
        save(ROOT/'reviews'/f'{hashlib.sha256(uid.encode()).hexdigest()[:16]}.json',row)
        return uid,row
    with ThreadPoolExecutor(2) as pool:
        futures=[]
        for uid,key,c in review_specs:
            started=time.monotonic();w=windows[key];short=copy.deepcopy(w['case']);short['interval']=[max(short['interval'][0],c['start']-4),min(short['interval'][1],max(c['end']+2,c['start']+4))]
            try:
                mix=render_mix(w['tracks'],short['interval'],ROOT/'audio'/f'review-{hashlib.sha256(uid.encode()).hexdigest()[:16]}.wav')
                t=time.monotonic();vote=judge.vote(short,c,mix,'en');audio_seconds=time.monotonic()-t
                futures.append(pool.submit(text_review,uid,key,c,vote,short,started,audio_seconds))
            except Exception as exc:
                reviews[uid]=dict(id=uid,window=key,first_label=initial[uid],review_complete=False,proposal=None,status='audio_failed',error=str(exc),seconds=time.monotonic()-started)
        for f in as_completed(futures):
            uid,row=f.result();reviews[uid]=row;record('EVENT_REVIEW_DONE',id=uid,status=row['status'],proposal=row['proposal'])
    final_results=copy.deepcopy(first_results)
    for uid in sorted(reviews,key=lambda x:all_candidates[x][1]['start']):
        row=reviews[uid];key,c=all_candidates[uid];result=final_results[key];proposal=row.get('proposal')
        if proposal is None or proposal==initial[uid]:continue
        judgment=copy.deepcopy(result.get('judgment',{'decisions':[]}))
        decision=next((d for d in judgment['decisions'] if d['candidate_id']==uid),None)
        if decision is None:row['status']='structural_rejection';row['error']='candidate_missing';continue
        decision.update(action='DROP' if proposal=='NEITHER' else 'KEEP',label='BACKCHANNEL' if proposal=='NEITHER' else proposal,merge_into='',reason='Automatic grounded two-view and local-audio consensus.')
        try:
            updated=finalize(windows[key]['case'],result['candidates'],judgment,result['speaker_gate']);updated['status']='ok'
            if label_of(updated,uid)!=proposal:raise ValueError('onset_gate_rejected_proposal')
            result.update(events=updated['events'],judgment=judgment,filtered_candidates=updated['filtered_candidates'])
            row['status']='changed'
        except Exception as exc:row.update(status='structural_rejection',error=str(exc))
    final={uid:label_of(final_results[key],uid) for uid,(key,c) in all_candidates.items()}
    review_wall=time.monotonic()-review_start;total_wall=time.monotonic()-first_start
    for row in event_state:
        uid=row['id'];row.update(final_label=final[uid],review_complete=reviews.get(uid,{}).get('review_complete',False),review_status=reviews.get(uid,{}).get('status','not_triggered'),review_seconds=reviews.get(uid,{}).get('seconds'),changed=initial[uid]!=final[uid])
    save(ROOT/'event_status.json',event_state);save(ROOT/'reviews.json',reviews);save(ROOT/'final_results.json',final_results)
    # References first enter scoring here; neither selection nor inference receives them.
    refs=json.loads(ref_path.read_text());evaluated=score(initial,final,reviews,refs)
    patterns={}
    for phase,results in [('first',first_results),('final',final_results)]:
        counts={}
        for cid,refrow in refs['recordings'].items():
            events={e['candidate_ids'][0]:e for key,r in results.items() if windows[key]['case']['conversation_id']==cid for e in r.get('events',[])}
            for p in refrow.get('patterns',[]):
                kind=p['kind'];counts.setdefault(kind,dict(total=0,correct=0));counts[kind]['total']+=1
                a,b=events.get(p['first']),events.get(p['second']);expected='INTERRUPT' if kind=='bidirectional' else 'BACKCHANNEL'
                good=bool(a and b and a['label']=='INTERRUPT' and b['label']==expected)
                if kind=='nested' and good:good=a['start']<=b['start']<a['end']
                counts[kind]['correct']+=int(good)
        patterns[phase]=counts
    evaluated['patterns']=patterns
    valid=sum(v in LABELS for v in final.values());completed=sum(r.get('review_complete',False) for r in reviews.values())
    summary=dict(status='complete',started_at=first_utc,finished_at=now(),reference_sha256=config['reference_sha256'],model_load_seconds=load_seconds,recordings_input=len(inputs),active_windows=len(windows),core_candidates=len(initial),valid_candidates=valid,review_selected=len(review_specs),review_completed=completed,review_failed=len(review_specs)-completed,review_rate=len(review_specs)/len(initial),review_completion=completed/len(review_specs) if review_specs else None,review_status_counts={s:sum(r['status']==s for r in reviews.values()) for s in sorted({r['status'] for r in reviews.values()})},first_wall_seconds=first_wall,review_wall_seconds=review_wall,with_review_wall_seconds=total_wall,first_targets_per_minute=60*sum(v in LABELS for v in initial.values())/first_wall,with_review_targets_per_minute=60*valid/total_wall,first_audio_requests=sum(len(r.get('audio_votes',[])) for r in first_results.values()),review_audio_requests=len(review_specs),first_text_requests=sum(len(r['attempts']) for r in first_results.values()),review_text_requests=sum(len(r.get(view,{}).get('attempts',[])) for r in reviews.values() for view in ['short','full']),scope=config['timing'],reference_agreement=evaluated['metrics'])
    save(ROOT/'evaluation.json',evaluated);save(ROOT/'summary.json',summary)
    record('EXPERIMENT_COMPLETE',summary=summary)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(ROOT/'fatal_error.json',dict(at=now(),error=str(exc),traceback=traceback.format_exc()));record('FATAL_ERROR',error=str(exc));raise
