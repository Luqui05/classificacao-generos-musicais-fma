"""Adaptador isolado para avaliar a CQT publicada na Librosa 0.5.0.

Lê somente funções numéricas de fontes locais com SHA-256 fixado. Não instala
nem substitui a Librosa atual. NumPy/SciPy e o interpolador resampy continuam
atuais; as adaptações de API e promoção de escalares são explícitas abaixo.
"""
import ast
import hashlib
from types import SimpleNamespace
import warnings

import librosa
import numpy as np
import scipy.fftpack
from librosa.util.deprecation import Deprecated
from librosa.util.exceptions import ParameterError


SOURCE_HASHES = {
    'constantq.py': '42b903fc420a6278252472f39a558f02dbd2de2ab139d384a5987e9a973ec2fb',
    'filters.py': '776c50892649b06fadda7093df45e1e8e0a4b87cd31f74a761bebb4c1bc5d79d',
}


class LegacyNumpy:
    """Aliases removidos e promoção de sqrt(escalar) compatível com NumPy 1.x."""
    def __getattr__(self, name):
        if name == 'float':
            return float
        if name == 'complex':
            return complex
        return getattr(np, name)

    @staticmethod
    def sqrt(value):
        result = np.sqrt(value)
        return result.item() if np.isscalar(result) else result


def load_functions(path, names, namespace):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[path.name]:
        raise ValueError(f'Fonte histórica diferente da versão fixada: {path.name}')
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError('Funções históricas ausentes')
    for node in functions:
        # Cache desativado; as expressões e operações numéricas permanecem originais.
        if any(not isinstance(d, ast.Call) or not isinstance(d.func, ast.Name) or d.func.id != 'cache'
               for d in node.decorator_list):
            raise ValueError('Decorador histórico não reconhecido')
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_legacy_cqt(source_dir, filter_dir=None):
    import resampy

    stft = librosa.stft
    util = SimpleNamespace(normalize=librosa.util.normalize, Deprecated=Deprecated,
        pad_center=lambda data, size, **kw: librosa.util.pad_center(data, size=size, **kw),
        sparsify_rows=librosa.util.sparsify_rows)
    np_legacy = LegacyNumpy()
    filter_namespace = dict(np=np_legacy, util=util, ParameterError=ParameterError,
        note_to_hz=librosa.note_to_hz, window_bandwidth=librosa.filters.window_bandwidth,
        get_window=librosa.filters.get_window)
    filters = load_functions(source_dir / 'filters.py',
        ['constant_q', 'constant_q_lengths', '__float_window'], filter_namespace)
    filters.window_bandwidth = librosa.filters.window_bandwidth

    def get_filter(name):
        if filter_dir is None:
            return resampy.filters.get_filter(name)
        with np.load(filter_dir / f'{name}.npz', allow_pickle=False) as archive:
            return archive['half_window'], int(archive['precision']), float(archive['rolloff'])

    def resample(y, sr_orig, sr_new, res_type='kaiser_fast', scale=False):
        # Reabre os coeficientes por chamada: o interpolador pode escalá-los.
        result = resampy.resample(y, sr_orig, sr_new, filter=lambda: get_filter(res_type))
        result = librosa.util.fix_length(result, size=int(np.ceil(len(y) * sr_new / sr_orig)))
        if scale:
            result /= float(np.sqrt(float(sr_new) / sr_orig))
        return result

    def legacy_stft(y, **kwargs):
        # A STFT 0.5.0 devolvia o conjugado da FFT, em complex64.
        return stft(y, **{**kwargs, 'pad_mode': 'reflect', 'dtype': np.complex64}).conj()

    audio = SimpleNamespace(resample=resample, BW_FASTEST=get_filter('kaiser_fast')[2])
    namespace = dict(np=np_legacy, fft=scipy.fftpack, util=util, filters=filters,
        audio=audio, warn=warnings.warn, ParameterError=ParameterError,
        cqt_frequencies=lambda n_bins, fmin, **kw: librosa.cqt_frequencies(n_bins, fmin=fmin, **kw),
        note_to_hz=librosa.note_to_hz,
        estimate_tuning=librosa.estimate_tuning, stft=legacy_stft)
    functions = load_functions(source_dir / 'constantq.py',
        ['cqt', '__cqt_filter_fft', '__trim_stack', '__cqt_response',
         '__early_downsample_count', '__early_downsample', '__num_two_factors'], namespace)
    return functions.cqt
