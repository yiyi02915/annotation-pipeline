import subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parent
base=Path('/data/L202500493/duplex_postprocess/runtime/glm')
with (root/'model_server.log').open('wb') as log:
    p=subprocess.Popen(['bash',str(base/'start_glm.sh')],cwd=base,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'model_server.pid').write_text(str(p.pid))
print(json.dumps(dict(pid=p.pid,log=str(root/'model_server.log'))))
