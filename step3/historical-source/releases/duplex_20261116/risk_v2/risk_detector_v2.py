"""Trainable error-risk sidecar. Uses predictions/target evidence, never reference labels at inference."""
import math
import re
import numpy as np

LABELS=['PAUSE','TURN_TRANSITION','INTERRUPT','BACKCHANNEL','NEITHER']
FEATURE_NAMES=['audio_disagrees','a_margin','b_margin','gap_seconds','target_seconds','target_word_count','left_word_count','right_word_count','question_near_onset','short_reply','other_activity','long_alignment_span','target_has_alignment_flags']+['a_'+k for k in LABELS]+['b_'+k for k in LABELS]


def features(row,a,b):
    p=row['input'];t=p['target'];ctx=p.get('context',[])
    lo,hi=t.get('absolute_gap',[t.get('start',0),t.get('end',0)])
    focus=p.get('focus',t.get('initiator'))
    target_text=t.get('text',t.get('right_word',''))
    words=lambda s:re.findall(r"[\w']+",s.lower())
    before=p.get('focus_before_target','');after=p.get('focus_after_target','')
    question=any('?' in s.get('text','') and s['start']<=lo and lo-2<=s['end']<=lo+.6 for s in ctx)
    other=any(s.get('speaker')!=focus and s['end']>=lo-.5 and s['start']<=hi for s in ctx)
    long_span=any((s['end']-s['start'])/max(1,len(words(s.get('text',''))))>2 for s in ctx)
    short=bool(re.fullmatch(r'(yes|yeah|yep|no|nope|right|okay|ok|sure|exactly|mhm|uh huh)[.!?, ]*',target_text.strip(),re.I))
    result=[float(a['predicted']!=b['predicted']),a.get('option_margin',0),b.get('option_margin',0),
            min(20,t.get('gap_seconds',0))/20,min(20,hi-lo)/20,min(30,len(words(target_text)))/30,
            min(30,len(words(before)))/30,min(30,len(words(after)))/30,float(question),float(short),float(other),float(long_span),float(bool(t.get('alignment_flags')))]
    result += [float(a['predicted']==c) for c in LABELS]+[float(b['predicted']==c) for c in LABELS]
    return result


def fit(x,y,penalty=2.0):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mean=x.mean(0);scale=np.maximum(x.std(0),.1)
    z=np.column_stack([np.ones(len(x)),(x-mean)/scale]);w=np.zeros(z.shape[1]);reg=np.eye(len(w))*penalty;reg[0,0]=1e-6
    weight=np.where(y==1,len(y)/(2*max(1,y.sum())),len(y)/(2*max(1,(1-y).sum())))
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(z@w,-30,30)))
        gradient=z.T@(weight*(p-y))+reg@w
        hessian=z.T@((weight*p*(1-p))[:,None]*z)+reg
        delta=np.linalg.solve(hessian,gradient);w-=delta
        if np.max(np.abs(delta))<1e-6:break
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist(),feature_names=FEATURE_NAMES,version='risk-v2-20261116',calibrated_probability=False)


def score(model,x):
    x=np.asarray(x,dtype=float);z=np.column_stack([np.ones(len(x)),(x-np.asarray(model['mean']))/np.asarray(model['scale'])])
    return 1/(1+np.exp(-np.clip(z@np.asarray(model['weights']),-30,30)))


def assess(flags,wrong):
    flags=np.asarray(flags,dtype=bool);wrong=np.asarray(wrong,dtype=bool)
    detected=int((flags&wrong).sum());count=int(flags.sum());errors=int(wrong.sum())
    return dict(total=len(flags),errors=errors,flagged=count,detected=detected,missed=errors-detected,false_flags=count-detected,
                error_recall=detected/errors if errors else None,flag_precision=detected/count if count else None,flag_rate=count/len(flags),
                retained_agreement=1-int((~flags&wrong).sum())/int((~flags).sum()) if (~flags).any() else None)
