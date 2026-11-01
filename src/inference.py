"""Inferência de MP3 com perfil FMA histórico e pipelines locais confiáveis."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import warnings

import joblib
from sklearn.utils.validation import check_is_fitted
from .audio_features import FEATURE_NAMES, extract_features, validate_features
from .models import build_models

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_models(directory):
    """Carrega apenas artefatos de origem confiável: hashes não autenticam joblib."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Treinamento incompleto')
    if manifest['features'] != list(FEATURE_NAMES):
        raise ValueError('Esquema incompatível no manifesto')
    if set(manifest['models']) != set(build_models()):
        raise ValueError('Esperados os cinco pipelines do experimento')
    for package, expected in manifest['versions'].items():
        if version(package) != expected:
            raise ValueError(f'Versão incompatível: {package}; esperada {expected}')
    paths = {}
    for name, entry in manifest['models'].items():
        path = (directory/entry['file']).resolve()
        if path.parent != directory or digest(path) != entry['sha256']:
            raise ValueError(f'Artefato inválido: {name}')
        paths[name] = path
    models = {}
    for name, path in paths.items():
        model = joblib.load(path)
        check_is_fitted(model.steps[-1][1])
        if list(model.feature_names_in_) != list(FEATURE_NAMES):
            raise ValueError(f'Colunas incompatíveis: {name}')
        if list(model.classes_) != manifest['classes']:
            raise ValueError(f'Classes incompatíveis: {name}')
        models[name] = model
    return models, manifest

def predict_features(features, models):
    vector = validate_features(features)
    frame = vector.to_frame().T
    return {name: str(model.predict(frame)[0]) for name, model in models.items()}

def infer_mp3(audio, model_directory, ffmpeg, output):
    audio, ffmpeg, output = Path(audio), Path(ffmpeg), Path(output)
    if output.exists():
        raise FileExistsError(f'Diretório já existe: {output}')
    if audio.suffix.lower() != '.mp3' or not audio.is_file():
        raise ValueError('Informe um arquivo MP3 existente')
    models, manifest = load_models(model_directory)
    audio_hash = digest(audio)
    ffmpeg_hash = digest(ffmpeg)
    ffmpeg_version = subprocess.run([str(ffmpeg.resolve()), '-version'], check=True,
                                    capture_output=True, text=True).stdout.splitlines()[0]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        features = extract_features(audio, profile='fma-2017', ffmpeg_path=ffmpeg)
        predictions = predict_features(features, models)
    if digest(audio) != audio_hash:
        raise RuntimeError('MP3 alterado durante a extração')
    result = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        audio_path=str(audio.resolve()), audio_sha256=audio_hash, profile='fma-2017',
        predictions=predictions, classes=manifest['classes'],
        model_directory=str(Path(model_directory).resolve()),
        model_manifest_sha256=digest(Path(model_directory)/'manifest.json'),
        model_sha256={name: entry['sha256'] for name,entry in manifest['models'].items()},
        ffmpeg_sha256=ffmpeg_hash, ffmpeg_version=ffmpeg_version,
        python=platform.python_version(),
        versions={name:version(name) for name in ('librosa','numpy','scipy','pandas','soundfile','numba','scikit-learn','joblib')},
        extractor_sha256=digest(Path(__file__).with_name('audio_features.py')),
        inference_sha256=digest(__file__), warnings=[str(w.message) for w in captured])
    output.mkdir(parents=True, exist_ok=False)
    features.to_csv(output/'features.csv', header=['value'], index_label='feature')
    result['features_sha256'] = digest(output/'features.csv')
    (output/'prediction.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
