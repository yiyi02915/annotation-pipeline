import json,hashlib
from pathlib import Path
import soundfile as sf
root=Path(__file__).resolve().parent;dest=root/'audio_evidence';dest.mkdir(exist_ok=True)
targets=json.loads((root/'alignment_evidence.json').read_text());out=[]
for t in targets:
    a,b=t['windows'][0];name=t['utterance']['utterance_id']+'.wav'
    with sf.SoundFile(t['audio']) as f:
        sr=f.samplerate;f.seek(int(a*sr));audio=f.read(int((b-a)*sr),dtype='float32')
    sf.write(dest/name,audio,sr,subtype='PCM_16')
    out.append(dict(parent_id=t['utterance']['utterance_id'],source_audio=t['audio'],source_window=[a,b],file='audio_evidence/'+name,sample_rate=sr,sha256=hashlib.sha256((dest/name).read_bytes()).hexdigest()))
(root/'audio_evidence_manifest.json').write_text(json.dumps(out,indent=2))
print('Exported',len(out),'audio evidence clips')
