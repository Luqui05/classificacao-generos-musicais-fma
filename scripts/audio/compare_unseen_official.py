"""Compara atributos oficiais e extraídos nas mesmas faixas e modelos salvos."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.audio_features import FEATURE_NAMES, validate_features
from src.audio_validation import DEFAULT_ATOL, DEFAULT_RTOL, read_official_features
from src.inference import digest, load_models


def compare(manifest_path, official_path, audio_run, model_directory, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 40 or manifest.track_id.duplicated().any():
        raise ValueError('Esperadas 40 faixas distintas')
    models, training = load_models(model_directory)
    labels = training['classes']
    if set(manifest.genre) != set(labels) or not manifest.groupby('genre').size().eq(5).all():
        raise ValueError('Esperadas cinco faixas de cada classe')
    if set(map(int, training['track_ids'])).intersection(manifest.track_id):
        raise ValueError('Sobreposição com treinamento')
    prior = json.loads((audio_run / 'run.json').read_text(encoding='utf-8'))
    manifest_hash = digest(manifest_path)
    model_hash = digest(model_directory / 'manifest.json')
    if (prior['status'] != 'complete' or prior['manifest_sha256'] != manifest_hash
            or prior['model_manifest_sha256'] != model_hash):
        raise ValueError('Execução de áudio incompatível com seleção/modelos atuais')
    official = read_official_features(official_path, manifest.track_id.tolist())
    extracted_rows = []
    numeric = []
    prior_predictions = pd.read_csv(audio_run / 'predictions.csv')
    expected_keys = {(int(i), name) for i in manifest.track_id for name in models}
    keys = list(zip(prior_predictions.track_id, prior_predictions.model))
    if len(keys) != 200 or set(keys) != expected_keys:
        raise ValueError('Predições anteriores ausentes ou duplicadas')
    prior_lookup = prior_predictions.set_index(['track_id', 'model'])
    for row in manifest.itertuples():
        folder = audio_run / f'{row.track_id:06d}'
        record = json.loads((folder / 'prediction.json').read_text(encoding='utf-8'))
        if (record['model_manifest_sha256'] != model_hash
                or record['features_sha256'] != digest(folder / 'features.csv')
                or record['model_sha256'] != {n: v['sha256'] for n, v in training['models'].items()}):
            raise ValueError(f'Proveniência incompatível: {row.track_id}')
        vector = pd.read_csv(folder / 'features.csv', index_col=0)['value']
        vector = validate_features(vector)
        extracted_rows.append(vector.to_numpy())
        close = np.isclose(vector.to_numpy(), official.loc[row.track_id].to_numpy(),
                           rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        numeric.append(dict(track_id=row.track_id, genre=row.genre,
                            matching_features=int(close.sum()), total_features=518))
        for name in models:
            previous = prior_lookup.loc[(row.track_id, name)]
            if previous.genre != row.genre or previous.prediction != record['predictions'][name]:
                raise ValueError('Rótulos ou predições anteriores inconsistentes')
    extracted = pd.DataFrame(extracted_rows, index=manifest.track_id, columns=FEATURE_NAMES)
    pairs, metrics, summary = [], {}, []
    truth = manifest.genre.to_numpy()
    for name, model in models.items():
        predicted = model.predict(official)
        replay = model.predict(extracted)
        previous = prior_lookup.xs(name, level='model').loc[manifest.track_id, 'prediction'].to_numpy()
        if not np.array_equal(replay, previous):
            raise ValueError(f'Reprodução das predições de áudio divergiu: {name}')
        metrics[name] = {}
        for variant, values in [('official', predicted), ('extracted', replay)]:
            report = classification_report(truth, values, labels=labels, output_dict=True, zero_division=0)
            report['accuracy'] = accuracy_score(truth, values)
            metrics[name][variant] = report
        audio_correct, official_correct = replay == truth, predicted == truth
        summary.append(dict(model=name,
                            f1_extracted=metrics[name]['extracted']['macro avg']['f1-score'],
                            f1_official=metrics[name]['official']['macro avg']['f1-score'],
                            accuracy_extracted=metrics[name]['extracted']['accuracy'],
                            accuracy_official=metrics[name]['official']['accuracy'],
                            changed_predictions=int((replay != predicted).sum()),
                            corrected_by_official=int((~audio_correct & official_correct).sum()),
                            worsened_by_official=int((audio_correct & ~official_correct).sum())))
        for track_id, genre, audio_prediction, official_prediction in zip(manifest.track_id, truth, replay, predicted):
            pairs.append(dict(track_id=int(track_id), genre=genre, model=name,
                              extracted_prediction=audio_prediction, official_prediction=official_prediction))
    # Nenhuma escrita nas entradas; saída nova somente após validar a comparação.
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(pairs).to_csv(output / 'paired_predictions.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(numeric).to_csv(output / 'feature_matches.csv', index=False)
    official.to_csv(output / 'official_features.csv', index_label='track_id')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    for name in models:
        selected = pd.DataFrame(pairs).query('model == @name')
        for variant in ('official', 'extracted'):
            pd.DataFrame(confusion_matrix(selected.genre, selected[f'{variant}_prediction'], labels=labels),
                         index=labels, columns=labels).to_csv(output / f'confusion_{name.replace(" ", "_")}_{variant}.csv', index_label='true_genre')
    provenance = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                      manifest_sha256=manifest_hash, model_manifest_sha256=model_hash,
                      model_sha256={n: v['sha256'] for n, v in training['models'].items()},
                      official_features_sha256=digest(official_path),
                      audio_run=str(audio_run.resolve()), audio_run_sha256=digest(audio_run / 'run.json'),
                      prior_predictions_sha256=digest(audio_run / 'predictions.csv'),
                      script_sha256=digest(Path(__file__)), count=40,
                      rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                      interpretation='paired feature-source comparison; does not isolate duration from decoder or extractor effects')
    (output / 'run.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Resultados: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('validation/unseen_fma/sample/manifest.csv'))
    parser.add_argument('--official', type=Path, default=Path('data/fma_metadata/features.csv'))
    parser.add_argument('--audio-run', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/unseen_official_comparison') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    compare(args.manifest, args.official, args.audio_run, args.models, args.output)


if __name__ == '__main__':
    main()
