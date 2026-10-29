"""Diagnósticos controlados dos 518 atributos da faixa completa 030520."""
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from diagnose_audio_rms import digest, historical_rms
from src.audio_features import FEATURE_NAMES, _feature_statistics, extract_features
from src.audio_validation import compare_vectors


PROFILES = {
    'current': dict(pcm16=False, historical_rms=False, stft_padding='constant'),
    'pcm16': dict(pcm16=True, historical_rms=False, stft_padding='constant'),
    'legacy_rms': dict(pcm16=True, historical_rms=True, stft_padding='constant'),
    'reflect_stft': dict(pcm16=True, historical_rms=True, stft_padding='reflect'),
    'ffmpeg_mp3float': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3float'),
    'ffmpeg_mp3': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3'),
    'chroma_bins12': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12),
    'reflect_cqt': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, cqt_padding='reflect'),
    'kaiser_cqt': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, cqt_padding='reflect', cqt_resampler='kaiser_fast'),
    'chroma_float64': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, cqt_padding='reflect', cqt_resampler='kaiser_fast', chroma_dtype='float64'),
    'legacy_cqt': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, chroma_dtype='float64', legacy_cqt=True),
    'legacy_cqt_filters': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, chroma_dtype='float64', legacy_cqt=True, old_filters=True),
    'legacy_cens': dict(pcm16=True, historical_rms=True, stft_padding='reflect', signal='ffmpeg_mp3', chroma_bins=12, chroma_dtype='float64', legacy_cqt=True, old_filters=True, periodic_cens=True),
}


def historical_zcr(y, frame_length=2048, hop_length=512):
    """Algoritmo 0.5.0: edge, threshold=1e-10, zero_pos=True e pad=False.

    Transcrição do cálculo NumPy; não importa nem executa código remoto.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError('Esperado áudio mono finito e não vazio')
    clipped = np.pad(y, frame_length // 2, mode='edge')
    clipped[np.abs(clipped) <= 1e-10] = 0
    frames = librosa.util.frame(clipped, frame_length=frame_length, hop_length=hop_length)
    signs = np.signbit(frames)
    crossings = np.pad(signs[:-1] != signs[1:], ((1, 0), (0, 0)), constant_values=False)
    return np.mean(crossings, axis=0, keepdims=True)


def extract_profile(audio, signal, sample_rate, parameters, legacy_cqt=None):
    """Reutiliza o extrator, aplicando substituições só dentro deste diagnóstico."""
    stft = librosa.stft
    cqt = librosa.cqt
    chroma_cqt = librosa.feature.chroma_cqt
    chroma_cens = librosa.feature.chroma_cens
    chroma_filters = librosa.filters.chroma

    def diagnostic_stft(*args, **kwargs):
        return stft(*args, **{**kwargs, 'pad_mode': parameters['stft_padding']})

    def diagnostic_cqt(*args, **kwargs):
        overrides = {}
        if 'cqt_padding' in parameters:
            overrides['pad_mode'] = parameters['cqt_padding']
        if 'cqt_resampler' in parameters:
            overrides['res_type'] = parameters['cqt_resampler']
        return cqt(*args, **{**kwargs, **overrides})

    def diagnostic_cens(**kwargs):
        overrides = dict(bins_per_octave=parameters['chroma_bins'])
        if parameters.get('periodic_cens'):
            # 0.5.0: scipy.signal.hanning(43, sym=False).
            overrides['smoothing_window'] = lambda n: librosa.filters.get_window('hann', n, fftbins=True)
        return chroma_cens(**{**kwargs, **overrides})

    with ExitStack() as context:
        context.enter_context(patch.object(librosa, 'load', return_value=(signal, sample_rate)))
        context.enter_context(patch.object(librosa, 'stft', side_effect=diagnostic_stft))
        if parameters['historical_rms']:
            context.enter_context(patch.object(librosa.feature, 'rms', side_effect=lambda **kw: historical_rms(kw['S'])))
        if 'chroma_bins' in parameters:
            context.enter_context(patch.object(librosa.feature, 'chroma_cqt', side_effect=lambda **kw: chroma_cqt(**{**kw, 'bins_per_octave': parameters['chroma_bins']})))
            context.enter_context(patch.object(librosa.feature, 'chroma_cens', side_effect=diagnostic_cens))
        if 'cqt_padding' in parameters or 'cqt_resampler' in parameters:
            context.enter_context(patch.object(librosa, 'cqt', side_effect=diagnostic_cqt))
        if 'chroma_dtype' in parameters:
            context.enter_context(patch.object(librosa.filters, 'chroma', side_effect=lambda **kw: chroma_filters(**{**kw, 'dtype': np.float64})))
        if parameters.get('legacy_cqt'):
            if legacy_cqt is None:
                raise ValueError('CQT histórica não disponibilizada')
            context.enter_context(patch.object(librosa, 'cqt', side_effect=legacy_cqt))
        return extract_features(audio)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, default=root / 'validation/audio_features/runs/full_030520_20261028T203735208700Z/030520_full.mp3')
    parser.add_argument('--baseline', type=Path, default=root / 'validation/audio_features/runs/20261028T192711309854Z')
    parser.add_argument('--ffmpeg', type=Path, default=root / 'validation/audio_features/runs/tools/imageio_ffmpeg_0_6_0/ffmpeg-win-x86_64-v7.1.exe')
    parser.add_argument('--resampy-path', type=Path, default=root / 'validation/audio_features/runs/tools/resampy_0_4_3', help='Pacote opcional extraído fora dos ambientes Python')
    parser.add_argument('--legacy-source', type=Path, default=root / 'validation/audio_features/runs/tools/librosa_0_5_0')
    parser.add_argument('--legacy-filters', type=Path, default=root / 'validation/audio_features/runs/tools/resampy_0_1_5')
    parser.add_argument('--profiles', choices=tuple(PROFILES), nargs='+', default=list(PROFILES))
    args = parser.parse_args()
    if args.resampy_path.is_dir():
        sys.path.insert(0, str(args.resampy_path.resolve()))
    cqt_adapters = {}
    if any(PROFILES[name].get('legacy_cqt') for name in args.profiles):
        from audio_legacy_diagnostics import load_legacy_cqt
        cqt_adapters[False] = load_legacy_cqt(args.legacy_source)
        cqt_adapters[True] = load_legacy_cqt(args.legacy_source, args.legacy_filters)
    table = pd.read_csv(args.baseline / 'differences.csv')
    selected = table.loc[table.track_id == 30520].set_index('feature')
    if len(selected) != 518 or set(selected.index) != set(FEATURE_NAMES):
        raise ValueError('Referência incompleta ou duplicada para 30520')
    reference = selected.official.reindex(FEATURE_NAMES)
    stereo, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    pcm16, sr16 = sf.read(args.audio, dtype='int16', always_2d=True)
    if sr != sr16:
        raise ValueError('Taxas inconsistentes entre leituras')
    signals = {'float32': np.mean(stereo.T, axis=0),
               'pcm16': np.mean(pcm16.astype(np.float32).T / 32768, axis=0)}
    decoder_evidence = {}
    if any(PROFILES[name].get('signal', '').startswith('ffmpeg_') for name in args.profiles):
        executable = args.ffmpeg.resolve()
        ffmpeg_version = subprocess.run([str(executable), '-version'], check=True, capture_output=True)
        decoder_evidence = dict(executable=str(executable), sha256=digest(executable),
                                version=ffmpeg_version.stdout.decode('utf-8', errors='replace'), commands={})
        for decoder in ('mp3float', 'mp3'):
            command = [str(executable), '-v', 'error', '-c:a', decoder, '-i', str(args.audio),
                       '-map', '0:a:0', '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1']
            result = subprocess.run(command, check=True, capture_output=True)
            channels = np.frombuffer(result.stdout, dtype='<i2').reshape(-1, stereo.shape[1])
            if len(channels) != len(stereo):
                raise ValueError('Contagem de amostras diferente no FFmpeg')
            signals['ffmpeg_' + decoder] = np.mean(channels.astype(np.float32).T / 32768, axis=0)
            decoder_evidence['commands'][decoder] = dict(command=command, stderr=result.stderr.decode('utf-8', errors='replace'))
    out = root / 'validation/audio_features/runs' / ('full_features_030520_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='running', track_id=30520, audio_path=str(args.audio.resolve()), audio_sha256=digest(args.audio),
                      baseline_run_sha256=digest(args.baseline / 'run.json'),
                      baseline_differences_sha256=digest(args.baseline / 'differences.csv'),
                      diagnostic_sha256=digest(Path(__file__)), extractor_sha256=digest(root / 'src/audio_features.py'),
                      sample_rate=sr, channels=stereo.shape[1], samples=len(signals['float32']),
                      duration_seconds=len(signals['float32']) / sr,
                      versions={name: version(name) for name in ('librosa', 'numpy', 'scipy', 'pandas', 'soundfile')},
                      python=platform.python_version(), libsndfile=sf.__libsndfile_version__,
                      tolerance=dict(rtol=1e-4, atol=1e-6), profiles={}, zcr_controls={},
                      decoders=decoder_evidence,
                      sources=['https://github.com/librosa/librosa/blob/0.5.0/librosa/core/audio.py',
                               'https://github.com/librosa/librosa/blob/0.5.0/librosa/feature/spectral.py'],
                      limitations=['Perfis isolam fatores; não reproduzem todo o ambiente de 2017.',
                                   'Uma faixa não valida a base inteira nem a inferência.'])
    if args.resampy_path.is_dir():
        provenance['versions']['resampy'] = version('resampy')
        provenance['resampy_source'] = json.loads((args.resampy_path / 'source.json').read_text(encoding='utf-8'))
    if cqt_adapters:
        provenance['legacy_cqt_sources'] = {p.name: digest(p) for p in args.legacy_source.glob('*.py')}
        provenance['legacy_cqt_adapter_sha256'] = digest(root / 'audio_legacy_diagnostics.py')
        provenance['legacy_filter_sources'] = {p.name: digest(p) for p in args.legacy_filters.glob('*.npz')}
        provenance['legacy_packages'] = [json.loads((directory / 'source.json').read_text(encoding='utf-8'))
                                         for directory in (args.legacy_source, args.legacy_filters)]
    zcr_comparisons = []
    for name, signal in signals.items():
        current = librosa.feature.zero_crossing_rate(signal)
        legacy = historical_zcr(signal)
        provenance['zcr_controls'][name] = dict(frames=current.shape[1],
            different_frames=int(np.count_nonzero(current != legacy)),
            max_frame_difference=float(np.max(np.abs(current - legacy))),
            waveform_sha256=hashlib.sha256(signal.tobytes()).hexdigest())
        for implementation, values in [('current', current), ('librosa_0_5_0', legacy)]:
            for stat, array in _feature_statistics(values).items():
                actual, expected = float(np.float32(array[0])), float(reference[f'zcr_{stat}_01'])
                error = abs(actual - expected)
                zcr_comparisons.append(dict(signal=name, implementation=implementation, statistic=stat,
                    official=expected, extracted=actual, absolute_error=error,
                    relative_error=error / abs(expected) if expected else None,
                    within_tolerance=error <= 1e-6 + 1e-4 * abs(expected)))
    pd.DataFrame(zcr_comparisons).to_csv(out / 'zcr_implementations.csv', index=False)
    all_details = []
    for profile in args.profiles:
        parameters = PROFILES[profile]
        signal = signals[parameters.get('signal', 'pcm16' if parameters['pcm16'] else 'float32')]
        signal_hash = hashlib.sha256(signal.tobytes()).hexdigest()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                vector = extract_profile(args.audio, signal, sr, parameters,
                                         legacy_cqt=cqt_adapters.get(parameters.get('old_filters', False)))
        except Exception as error:
            provenance['status'] = 'failed'
            provenance['profiles'][profile] = dict(parameters=parameters, error=str(error))
            (out / 'diagnostic.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            if all_details:
                pd.concat(all_details, ignore_index=True).to_csv(out / 'differences.csv', index=False)
            raise
        if profile == 'current':
            # Controle real da injeção: mesmo resultado do extrator sem substituições.
            np.testing.assert_array_equal(vector.to_numpy(), extract_features(args.audio).to_numpy())
            provenance['injection_control_matches_unmodified_extractor'] = True
        if hashlib.sha256(signal.tobytes()).hexdigest() != signal_hash:
            raise RuntimeError('Perfil modificou as amostras de entrada')
        details = compare_vectors(reference, vector)
        details.insert(0, 'profile', profile)
        all_details.append(details)
        vector.to_csv(out / f'{profile}_features.csv')
        family = details.groupby('family').within_tolerance.agg(['sum', 'count'])
        provenance['profiles'][profile] = dict(parameters=parameters, within_tolerance=int(details.within_tolerance.sum()),
            comparisons=len(details), input_sha256=signal_hash,
            max_tolerance_ratio=float((details.absolute_error / (1e-6 + 1e-4 * details.official.abs())).max()),
            families={k: dict(within_tolerance=int(v['sum']), comparisons=int(v['count']))
                                               for k, v in family.iterrows()},
            warnings=[dict(category=type(w.message).__name__, message=str(w.message)) for w in captured])
        (out / 'diagnostic.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(profile, provenance['profiles'][profile]['within_tolerance'], '/ 518')
        print(family.to_string())
    combined = pd.concat(all_details, ignore_index=True)
    combined.to_csv(out / 'differences.csv', index=False)
    combined.groupby(['profile', 'family']).within_tolerance.agg(['sum', 'count']).to_csv(out / 'family_summary.csv')
    provenance['status'] = 'complete'
    (out / 'diagnostic.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
