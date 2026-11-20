import hashlib,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
variants=['dedup','anchored','teacher','teacher_targeted','teacher_targeted_strong']
for name in variants:assert (root/name/'complete.json').exists(), name+' not complete'
with tarfile.open(root/'quality_results.tgz','w:gz') as archive:
    for name in variants:
        for path in sorted((root/name).glob('*.json')):archive.add(path,arcname=str(path.relative_to(root)))
    archive.add(root/'teacher_targeted_strong/selected_lora',arcname='teacher_targeted_strong/selected_lora')
    for pattern in ['*.py','*.log','teacher.json','teacher_targeted.json','diagnostic*.json']:
        for path in sorted(root.glob(pattern)):archive.add(path,arcname=path.name)
print(hashlib.sha256((root/'quality_results.tgz').read_bytes()).hexdigest())
