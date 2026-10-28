"""Diagnóstico isolado de RMS; não altera o extrator nem execuções anteriores."""
from pathlib import Path
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from unittest.mock import patch

import audioread
import librosa
import numpy as np
import soundfile as sf

from src.audio_features import _feature_statistics


def historical_rms(spectrum):
    """Ramo S da feature.rmse da Librosa 0.5.0, sem executar código remoto."""
    return np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0, keepdims=True))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    baseline = root / 'validation/audio_features/runs/20261028T192711309854Z'
    audio = root / 'data/fma_small/030/030520.mp3'
    with (baseline / 'differences.csv').open(newline='', encoding='utf-8') as stream:
        reference = {r['feature']: float(r['official']) for r in csv.DictReader(stream)
                     if int(r['track_id']) == 30520}
    if len(reference) != 518:
        raise ValueError('Referência incompleta para 30520')
    stereo, sr = sf.read(audio, dtype='float32', always_2d=True)
    pcm16, sr16 = sf.read(audio, dtype='int16', always_2d=True)
    # Observa a chamada efetiva ao SoundFile durante librosa.load.
    import librosa.core.audio as audio_module
    with patch.object(audio_module, '__soundfile_load', wraps=audio_module.__soundfile_load) as observed:
        loaded, loaded_sr = librosa.load(audio, sr=None, mono=True)
    mono = np.mean(stereo.T, axis=0)
    quantized = np.mean((pcm16.astype(np.float32) / 32768).T, axis=0)
    assert sr == sr16 == loaded_sr
    assert np.array_equal(loaded, mono)
    decoder = {'available_backends': [c.__module__ + '.' + c.__name__
                                    for c in audioread.available_backends()],
               'librosa_soundfile_calls': observed.call_count,
               'librosa_equals_soundfile_mono': True}
    signals = {'soundfile_float32': mono, 'soundfile_pcm16': quantized}
    try:
        with audioread.audio_open(str(audio)) as handle:
            decoder['audioread_backend'] = type(handle).__module__ + '.' + type(handle).__name__
            blocks = [np.frombuffer(block, dtype='<i2').astype(np.float32) / 32768
                      for block in handle]
            decoded = np.concatenate(blocks).reshape(-1, handle.channels).T
            if handle.samplerate != sr:
                raise ValueError('Taxa diferente; diagnóstico não faz reamostragem')
            signals['audioread_pcm16'] = np.mean(decoded, axis=0)
    except audioread.NoBackendError:
        decoder['audioread_error'] = 'NoBackendError: nenhum backend disponível decodifica este MP3'
    rows = []
    for label, y in signals.items():
        for padding in ('constant', 'reflect'):
            spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512, pad_mode=padding))
            for formula, values in [('modern', librosa.feature.rms(S=spectrum)),
                                    ('librosa_0_5_0_formula', historical_rms(spectrum))]:
                for stat, array in _feature_statistics(values).items():
                    value = float(np.float32(array[0]))
                    official = reference[f'rmse_{stat}_01']
                    error = abs(value - official)
                    rows.append(dict(signal=label, padding=padding, formula=formula,
                                     statistic=stat, official=official, extracted=value,
                                     absolute_error=error,
                                     relative_error=error / abs(official) if official else None,
                                     within_tolerance=bool(error <= 1e-6 + 1e-4 * abs(official))))
    controls = {}
    for label, y in signals.items():
        zcr = _feature_statistics(librosa.feature.zero_crossing_rate(y))
        controls[label] = {'samples': len(y), 'duration_seconds': len(y) / sr,
                           'waveform_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                           'time_rms': float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
                           'zcr_mean': float(zcr['mean'][0])}
    delta = mono.astype(np.float64) - quantized.astype(np.float64)
    report = dict(track_id=30520, audio_sha256=digest(audio),
                  baseline_run_sha256=digest(baseline / 'run.json'),
                  baseline_differences_sha256=digest(baseline / 'differences.csv'),
                  script_sha256=digest(Path(__file__)), sample_rate=sr, channels=stereo.shape[1],
                  versions=dict(python=platform.python_version(), librosa=librosa.__version__,
                                numpy=np.__version__, soundfile=sf.__version__,
                                libsndfile=sf.__libsndfile_version__),
                  decoder=decoder, signals=controls,
                  pcm16_max_absolute_sample_error=float(np.max(np.abs(delta))),
                  pcm16_rmse_sample_error=float(np.sqrt(np.mean(delta ** 2))),
                  official_zcr_mean=reference['zcr_mean_01'],
                  limitations=['PCM16 SoundFile usa o mesmo decodificador; não substitui FFmpeg.',
                               'A fórmula histórica usa a STFT e as estatísticas do ambiente atual.',
                               'Não demonstra que o trecho de áudio coincide com o usado no CSV.'],
                  historical_source='https://github.com/librosa/librosa/blob/0.5.0/librosa/feature/spectral.py')
    out = root / 'validation/audio_features/runs' / ('rms_030520_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'rms_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'diagnostic.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        if row['statistic'] == 'mean':
            print(row)


if __name__ == '__main__':
    main()
