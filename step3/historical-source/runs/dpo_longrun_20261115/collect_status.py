import json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text())
plan=read(root/'plan.json');jobs=[]
for cfg in plan['jobs']:
    path=root/'jobs'/cfg['id'];done=read(path/'complete.json') if (path/'complete.json').exists() else read(path/'status.json')
    checks=[json.loads(l) for l in (path/'checkpoints.jsonl').read_text().splitlines()] if (path/'checkpoints.jsonl').exists() else []
    best=done.get('best_validation',{})
    jobs.append(dict(id=cfg['id'],status=done['status'],reason=done.get('reason'),steps=done.get('step'),
        completed_epochs=sum(r['reason']=='epoch_end' for r in checks),checkpoints=len(checks),
        validation_correct=best.get('correct'),validation_total=best.get('total'),macro_recall=done.get('best_score'),
        best_path=done.get('best_path'),elapsed_seconds=done.get('elapsed_seconds'),finished=done.get('time'),
        latest_checkpoint=checks[-1] if checks else None))
events=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
summary=dict(supervisor=read(root/'supervisor.json'),final_status=read(root/'final_status.json') if (root/'final_status.json').exists() else None,
    jobs=jobs,total_checkpoints=sum(r['checkpoints'] for r in jobs),retries=[e for e in events if e['event']=='retry'])
(root/'completion_summary.json').write_text(json.dumps(summary,indent=2))
with tarfile.open(root/'completion_records.tgz','w:gz') as archive:
    for name in ['completion_summary.json','progress.json','supervisor.json','selection.json','final_status.json','final_report.txt','final_metrics.json','resume_test.json','events.jsonl','supervisor.log','final_evaluation.log']:
        if (root/name).exists():archive.add(root/name,arcname=name)
    for cfg in plan['jobs']:
        for name in ['complete.json','config.json','status.json','latest.json','checkpoints.jsonl','events.jsonl','pair_audit.json','train.log']:
            path=root/'jobs'/cfg['id']/name
            if path.exists():archive.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(summary,indent=2))
