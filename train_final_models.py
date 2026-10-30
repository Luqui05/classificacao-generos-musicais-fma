"""Ajusta os cinco pipelines sobre a base completa deduplicada e os persiste."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import warnings

import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from src.data_loader import load_dataset
from src.models import build_models

ROOT = Path(__file__).resolve().parent

def assert_same_state(left, right):
    """Compara estado lógico, ignorando padding de structs internos da árvore."""
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_state(a, b)
    elif hasattr(left, '__dict__'):
        assert type(left) is type(right)
        assert_same_state(vars(left), vars(right))
    elif type(left).__module__.startswith('sklearn.'):
        assert type(left) is type(right)
        assert_same_state(left.__getstate__(), right.__getstate__())
    else:
        np.testing.assert_equal(left, right)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def check_environment(lock):
    versions = {}
    for line in Path(lock).read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f'Ambiente incompatível: {name}={actual}; lock exige {expected}')
        versions[name] = actual
    return versions

def train(csv_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Diretório já existe: {output}')
    versions = check_environment(ROOT/'requirements-lock.txt')
    protected = [ROOT/'requirements-lock.txt', ROOT/'src/models.py', ROOT/'src/data_loader.py',
                 ROOT/'main.py', ROOT/'src/cross_validation.py', Path(csv_path),
                 *[p for p in (ROOT/'results').rglob('*') if p.is_file()]]
    before = {str(p.resolve()): digest(p) for p in protected}
    bundle = load_dataset(csv_path, drop_duplicate_samples=True)
    if bundle.X.shape != (7984, 518):
        raise ValueError(f'Dimensões inesperadas para o experimento: {bundle.X.shape}')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='final_fit_all_deduplicated_samples_no_evaluation', python=platform.python_version(),
        versions=versions, diagnostics=bundle.diagnostics, features=list(bundle.X.columns),
        classes=list(bundle.class_names), track_ids=bundle.track_ids.tolist(),
        class_counts={str(k):int(v) for k,v in bundle.y.value_counts().items()},
        protected_sha256=before, trainer_sha256=digest(__file__), models={})
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False),encoding='utf-8')
    save_manifest()
    try:
        for index, (name, model) in enumerate(build_models().items(), 1):
            print(f'Treinando {name}...', flush=True)
            parameters = {step: estimator.get_params(deep=False) for step, estimator in model.steps}
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(bundle.X, bundle.y)
            seconds = time.perf_counter()-start
            check_is_fitted(model.steps[-1][1])
            path = output/f'{index:02d}_{name.lower().replace(" ", "_")}.joblib'
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)  # somente artefato recém-criado e confiável
            check_is_fitted(restored.steps[-1][1])
            assert_same_state(model, restored)
            if list(restored.feature_names_in_) != list(bundle.X.columns):
                raise RuntimeError('Esquema alterado na serialização')
            manifest['models'][name] = dict(file=path.name, sha256=digest(path),
                parameters=parameters, fit_seconds=seconds, serialization_verified=True,
                warnings=[str(w.message) for w in captured])
            save_manifest()
        if any(digest(p) != sha for p,sha in before.items()):
            raise RuntimeError('Arquivo protegido mudou durante o treinamento')
        manifest['status'] = 'complete'
        manifest['protected_files_unchanged'] = True
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        save_manifest()
    print(output.resolve(), flush=True)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/fma_small_features.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/final_models'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    train(args.data, args.output)
