import os, json, time
from pathlib import Path
ROOT=Path('/data/L202500493/duplex_postprocess')
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, snapshot_download
repo='adventists-ai/DuplexJev-4B-Para'
info=HfApi().model_info(repo)
dest=ROOT/'models/DuplexJev-4B-Para'
dest.mkdir(parents=True,exist_ok=True)
(dest/'download_provenance.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'started':time.time()},indent=2))
print('Downloading',repo,info.sha,flush=True)
snapshot_download(repo_id=repo,revision=info.sha,local_dir=dest,max_workers=4)
print('DOWNLOAD_COMPLETE',flush=True)
