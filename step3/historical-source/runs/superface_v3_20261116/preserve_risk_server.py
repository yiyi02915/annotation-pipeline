import hashlib
import json
import shutil
from pathlib import Path

base = Path('/data/L202500493/duplex_postprocess')
run = base / 'runs/superface_v3_20261116'
root = base / 'releases/duplex_20261116/risk_v2'
root.mkdir(parents=True, exist_ok=True)
for task in ['gap', 'overlap']:
    target = root / ('auxiliary_' + task)
    target.mkdir(exist_ok=True)
    for source in (run / f'abc/B/{task}/epoch_1').iterdir():
        if source.is_file():
            shutil.copy2(source, target / source.name)
config = dict(
    scope='best_sft_A only; not calibrated for original 27B pipeline',
    main_adapters={'gap': '../best_sft/gap', 'overlap': '../best_sft/overlap'},
    auxiliary_adapters={'gap': 'auxiliary_gap', 'overlap': 'auxiliary_overlap'},
    semantics='Preserve main predicted labels; flag for automatic recheck separately',
    calibrated_probability=False,
    files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'})
(root / 'manifest.json').write_text(json.dumps(config, indent=2))
print(json.dumps({'risk_release': str(root), 'files': len(config['files'])}))
