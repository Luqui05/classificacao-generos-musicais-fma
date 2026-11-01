"""Experimento isolado: transcrição do interpolador resampy 0.1.5, mono.

Executar da raiz com --track-id e --audio; requer ferramentas locais em runs/.
Não modifica o extrator. Não é implementação destinada à inferência.
"""
from pathlib import Path
import argparse
import hashlib, json
import sys
import numpy as np
from numba import njit
root=Path.cwd(); sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'validation/audio_features/runs/tools/resampy_0_4_3'))
import resampy
@njit
def interpolate(x,y,ratio,w,d,precision):
    scale=min(1.,ratio); step=int(scale*precision); time=0.
    for t in range(len(y)):
        n=int(time); frac=scale*(time-n); ix=frac*precision; offset=int(ix); eta=ix-offset
        i=0
        for i in range(min(n+1,(len(w)-offset)//step)):
            y[t] += (w[offset+i*step]+eta*d[offset+i*step])*x[n-i]
        frac=scale-frac; ix=frac*precision; offset=int(ix); eta=ix-offset
        for k in range(min(len(x)-i-n+1,(len(w)-offset)//step)):
            if n+k+1 >= len(x): raise ValueError('Unsafe historical index')
            y[t] += (w[offset+k*step]+eta*d[offset+k*step])*x[n+k+1]
        time+=1./ratio
def resample(x,sr_orig,sr_new,filter):
    ratio=float(sr_new)/sr_orig; y=np.zeros(int(len(x)*ratio),dtype=x.dtype)
    w,p,_=filter()
    if ratio<1: w*=ratio
    d=np.zeros_like(w); d[:-1]=np.diff(w)
    interpolate(x,y,ratio,w,d,p)
    return y
resampy.resample=resample
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--track-id',type=int,required=True)
parser.add_argument('--audio',type=Path,required=True)
args=parser.parse_args()
source=(root/'scripts/audio/diagnose_full_audio.py').read_text(encoding='utf-8').replace('30520',str(args.track_id)).replace('030520',f'{args.track_id:06d}')
sys.argv=['diagnose_full_audio.py','--audio',str(args.audio),'--profiles','legacy_cens']
runs=root/'validation/audio_features/runs'
before=set(runs.glob('full_features_*'))
exec(compile(source,str(root/'scripts/audio/diagnose_full_audio.py'),'exec'),{'__name__':'__main__','__file__':str(root/'scripts/audio/diagnose_full_audio.py')})
for output in set(runs.glob('full_features_*'))-before:
    (output/'interpolator_override.json').write_text(json.dumps(dict(
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapted_diagnostic_sha256=hashlib.sha256(source.encode()).hexdigest(),
        interpolator='transcription of resampy 0.1.5 interp.pyx using numba, mono, bounds guard',
        source_sha256='7225c928bd730fb4e7b03d4c8df9e516093159f368dd2e8395ff4aa5e00fe2e2',
        limitations='diagnostic.json identifies original files; this sidecar records runtime overrides'
    ),indent=2),encoding='utf-8')
