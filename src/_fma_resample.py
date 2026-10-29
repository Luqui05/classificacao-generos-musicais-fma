"""Transcrição mono do interpolador resampy 0.1.5 (licença em fma_2017/).

Preserva o limite histórico da asa direita. Não usar como resampler genérico.
"""
import numpy as np
from numba import njit

@njit
def interpolate(x, y, ratio, w, d, precision):
    scale = min(1., ratio)
    step = int(scale * precision)
    if step < 1:
        raise ValueError('Razão de reamostragem histórica não suportada')
    time = 0.
    for t in range(len(y)):
        n = int(time)
        frac = scale * (time - n)
        ix = frac * precision
        offset = int(ix)
        eta = ix - offset
        i = 0
        for i in range(min(n + 1, (len(w) - offset) // step)):
            y[t] += (w[offset + i*step] + eta*d[offset + i*step])*x[n-i]
        frac = scale - frac
        ix = frac * precision
        offset = int(ix)
        eta = ix - offset
        for k in range(min(len(x) - i - n + 1, (len(w) - offset) // step)):
            if n+k+1 >= len(x):
                raise ValueError('Índice histórico fora dos limites')
            y[t] += (w[offset + k*step] + eta*d[offset + k*step])*x[n+k+1]
        time += 1./ratio

def resample(x, sr_orig, sr_new, filter):
    if x.ndim != 1 or not np.isfinite(x).all() or sr_orig <= 0 or sr_new <= 0:
        raise ValueError('Esperado áudio mono finito e taxas positivas')
    ratio = float(sr_new)/sr_orig
    y = np.zeros(int(len(x)*ratio), dtype=x.dtype)
    if not y.size:
        raise ValueError('Áudio curto demais para reamostragem histórica')
    w, precision, _ = filter()
    if ratio < 1:
        w *= ratio
    delta = np.zeros_like(w)
    delta[:-1] = np.diff(w)
    interpolate(x, y, ratio, w, delta, precision)
    return y
