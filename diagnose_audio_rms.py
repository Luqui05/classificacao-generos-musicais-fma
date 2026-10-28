"""Diagnóstico isolado de RMS; não altera o extrator nem execuções anteriores."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from unittest.mock import patch

import audioread
import audioread.ffdec
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from src.audio_features import _feature_statistics


def historical_rms(spectrum):
    """Ramo S da feature.rmse da Librosa 0.5.0, sem executar código remoto."""
    return np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0, keepdims=True))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_waveforms(reference, candidate, max_lag=2048):
    """Compara antes/depois de alinhar; lag positivo remove amostras da referência."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.ndim != 1 or not reference.size or not candidate.size:
        raise ValueError('Esperados dois vetores mono não vazios')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Amostras não finitas')
    if max_lag < 0:
        raise ValueError('max_lag deve ser não negativo')
    count = min(reference.size, candidate.size)
    raw_delta = reference[:count] - candidate[:count]
    # Dez segundos bastam para investigar atraso de até 2048 amostras nesta faixa.
    window = min(count, 441000)
    corr = correlate(reference[:window], candidate[:window], method='fft')
    lags = correlation_lags(window, window)
    allowed = np.abs(lags) <= max_lag
    lag = int(lags[allowed][np.argmax(corr[allowed])]) if np.any(corr[allowed]) else 0
    start_ref, start_candidate = max(0, lag), max(0, -lag)
    overlap = min(reference.size - start_ref, candidate.size - start_candidate)
    if overlap <= 0:
        raise ValueError('Não há sobreposição após alinhamento')
    a = reference[start_ref:start_ref + overlap]
    b = candidate[start_candidate:start_candidate + overlap]
    delta = a - b
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(reference_samples=reference.size, candidate_samples=candidate.size,
                sample_count_difference=int(candidate.size - reference.size),
                raw_rmse=float(np.sqrt(np.mean(raw_delta ** 2))),
                lag_samples=lag, max_lag_samples=max_lag, overlap_samples=overlap,
                aligned_rmse=float(np.sqrt(np.mean(delta ** 2))),
                aligned_max_absolute_error=float(np.max(np.abs(delta))),
                aligned_cosine_similarity=float(np.dot(a, b) / norm) if norm else None,
                exactly_equal=bool(np.array_equal(reference, candidate)))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', type=Path, help='Executável portátil; não altera o PATH')
    parser.add_argument('--audio', type=Path, help='Outra versão da faixa 030520; mantém a referência desse ID')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    baseline = root / 'validation/audio_features/runs/20261028T192711309854Z'
    audio = args.audio.resolve() if args.audio else root / 'data/fma_small/030/030520.mp3'
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
    ffmpeg = args.ffmpeg.resolve() if args.ffmpeg else None
    if ffmpeg:
        if not ffmpeg.is_file():
            raise FileNotFoundError(ffmpeg)
        ffmpeg_version = subprocess.run([str(ffmpeg), '-version'], capture_output=True, check=True)
        command = [str(ffmpeg), '-v', 'error', '-i', str(audio), '-map', '0:a:0',
                   '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
        decoded = subprocess.run(command, capture_output=True, check=True)
        channels = np.frombuffer(decoded.stdout, dtype='<f4').reshape(-1, stereo.shape[1]).T
        signals['ffmpeg_float32'] = np.mean(channels, axis=0)
        decoder['ffmpeg'] = dict(path=str(ffmpeg), sha256=digest(ffmpeg),
                                version=ffmpeg_version.stdout.decode('utf-8', errors='replace'),
                                float32_command=command,
                                float32_stderr=decoded.stderr.decode('utf-8', errors='replace'))
        package_source = ffmpeg.parent / 'source.json'
        if package_source.is_file():
            decoder['ffmpeg']['package_source'] = json.loads(package_source.read_text(encoding='utf-8'))
    try:
        # Patch local ao diagnóstico: força o caminho escolhido apenas no audioread.
        commands = (str(ffmpeg),) if ffmpeg else audioread.ffdec.COMMANDS
        with patch.object(audioread.ffdec, 'COMMANDS', commands):
            handle = audioread.ffdec.FFmpegAudioFile(str(audio)) if ffmpeg else audioread.audio_open(str(audio))
        with handle:
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
    zcr_rows = []
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
        for stat, array in zcr.items():
            value = float(np.float32(array[0]))
            official = reference[f'zcr_{stat}_01']
            error = abs(value - official)
            zcr_rows.append(dict(signal=label, statistic=stat, official=official,
                                 extracted=value, absolute_error=error,
                                 relative_error=error / abs(official) if official else None,
                                 within_tolerance=bool(error <= 1e-6 + 1e-4 * abs(official))))
        controls[label] = {'samples': len(y), 'duration_seconds': len(y) / sr,
                           'waveform_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                           'time_rms': float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
                           'zcr_mean': float(zcr['mean'][0])}
    delta = mono.astype(np.float64) - quantized.astype(np.float64)
    report = dict(track_id=30520, audio_path=str(audio), audio_sha256=digest(audio),
                  baseline_run_sha256=digest(baseline / 'run.json'),
                  baseline_differences_sha256=digest(baseline / 'differences.csv'),
                  script_sha256=digest(Path(__file__)), sample_rate=sr, channels=stereo.shape[1],
                  versions=dict(python=platform.python_version(), librosa=librosa.__version__,
                                audioread=version('audioread'),
                                numpy=np.__version__, soundfile=sf.__version__,
                                libsndfile=sf.__libsndfile_version__),
                  decoder=decoder, signals=controls,
                  waveform_comparisons={label: compare_waveforms(mono, y)
                                        for label, y in signals.items() if label != 'soundfile_float32'},
                  source_peak=float(np.max(np.abs(stereo))),
                  source_values_outside_pcm16=int(np.count_nonzero((stereo < -1) | (stereo > 32767 / 32768))),
                  pcm16_max_absolute_sample_error=float(np.max(np.abs(delta))),
                  pcm16_rmse_sample_error=float(np.sqrt(np.mean(delta ** 2))),
                  official_zcr_mean=reference['zcr_mean_01'],
                  limitations=['PCM16 SoundFile usa o mesmo decodificador; não substitui FFmpeg.',
                               'FFmpeg atual não reproduz por si só o decodificador histórico da FMA.',
                               'A fórmula histórica usa a STFT e as estatísticas do ambiente atual.',
                               'Não demonstra que o trecho de áudio coincide com o usado no CSV.'],
                  historical_source='https://github.com/librosa/librosa/blob/0.5.0/librosa/feature/spectral.py')
    out = root / 'validation/audio_features/runs' / ('rms_030520_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'rms_comparison.csv', rows)
    write_csv(out / 'zcr_comparison.csv', zcr_rows)
    (out / 'diagnostic.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        if row['statistic'] == 'mean':
            print(row)


if __name__ == '__main__':
    main()
