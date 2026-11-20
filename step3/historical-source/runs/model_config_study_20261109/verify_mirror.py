import os,json
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import HfApi
D=Path('/data/L202500493/duplex_postprocess/runs/model_config_study_20261109');ms=json.loads((D/'modelscope_manifest.json').read_text());m={f['Name']:f for f in ms['Data']['Files']}
info=HfApi().model_info('adventists-ai/DuplexJev-32B-Para',revision='757f9df3bf72ffc5b7a708696d47982c39d48d8c',files_metadata=True)
files=[]
for f in info.siblings:
 if f.rfilename.endswith('.safetensors'):
  assert f.lfs.sha256==m[f.rfilename]['Sha256'],f.rfilename
  files.append(dict(name=f.rfilename,size=f.size,sha256=f.lfs.sha256,ms_revision=m[f.rfilename]['Revision']))
(D/'verified_mirror_files.json').write_text(json.dumps(files,indent=2));print('MATCHING_HASHES',len(files),sum(f['size'] for f in files))
