"""Avalia os cinco pipelines salvos em um manifesto de faixas não treinadas."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.inference import digest, infer_mp3, load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('validation/unseen_fma/sample/manifest.csv'))
    parser.add_argument('--audio-root', type=Path, default=Path('data/fma_medium'))
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/unseen_evaluation') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = pd.read_csv(args.manifest)
    _, training = load_models(args.models)
    labels = training['classes']
    if manifest.track_id.duplicated().any() or not set(manifest.genre).issubset(labels):
        raise ValueError('IDs duplicados ou gêneros inválidos no manifesto')
    # A seleção exclui toda a Small; conferir novamente os IDs de treinamento.
    training_ids = set(map(int, training['track_ids']))
    if training_ids.intersection(map(int, manifest.track_id)):
        raise ValueError('Faixas do treinamento no conjunto de avaliação')
    root = args.audio_root.resolve()
    audios = []
    for row in manifest.itertuples():
        audio = (root / row.relative_path).resolve()
        if not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError(f'Caminho inválido ou ausente: {audio}')
        audios.append(audio)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='running', manifest_sha256=digest(args.manifest),
                      model_manifest_sha256=digest(args.models / 'manifest.json'),
                      scope='unseen tracks within FMA', profile='fma-2017', count=len(manifest))
    def save():
        (args.output / 'run.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    predictions = []
    try:
        for index, (row, audio) in enumerate(zip(manifest.itertuples(), audios), 1):
            result = infer_mp3(audio, args.models, args.ffmpeg, args.output / f'{row.track_id:06d}')
            for model, predicted in result['predictions'].items():
                predictions.append(dict(track_id=row.track_id, genre=row.genre, model=model, prediction=predicted))
            print(f'[{index}/{len(manifest)}] {row.track_id:06d}', flush=True)
        frame = pd.DataFrame(predictions)
        frame.to_csv(args.output / 'predictions.csv', index=False)
        metrics = {}
        for model, group in frame.groupby('model'):
            report = classification_report(group.genre, group.prediction, labels=labels, output_dict=True, zero_division=0)
            report['accuracy'] = accuracy_score(group.genre, group.prediction)
            metrics[model] = report
            pd.DataFrame(confusion_matrix(group.genre, group.prediction, labels=labels), index=labels, columns=labels).to_csv(args.output / f'confusion_{model.replace(" ", "_")}.csv', index_label='true_genre')
        (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        provenance['status'] = 'complete'
    except Exception as error:
        provenance.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print(f'Resultados: {args.output.resolve()}')


if __name__ == '__main__':
    main()
