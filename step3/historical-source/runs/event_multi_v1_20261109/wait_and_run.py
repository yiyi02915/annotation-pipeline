import time,urllib.request,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
deadline=time.time()+900
while time.time()<deadline:
 try:
  with urllib.request.urlopen('http://127.0.0.1:18770/v1/models',timeout=3) as r:
   if r.status==200:break
 except Exception:time.sleep(5)
else:raise TimeoutError('Qwen service startup timeout')
subprocess.run([sys.executable,'-u',str(D/'run_multi.py')],check=True)
