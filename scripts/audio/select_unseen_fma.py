"""Seleciona faixas FMA fora da Small; não carrega nem executa modelos.

Usa somente a biblioteca padrão. A ordenação SHA-256 independe da versão do
gerador aleatório: para cada gênero ordena hash('seed:genre:track_id').
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
GENRES = ('Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental',
          'International', 'Pop', 'Rock')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select(tracks_path, features_path, output, per_genre=5, seed=42):
    if output.exists():
        raise FileExistsError(f'A saída já existe: {output}')
    if per_genre < 1:
        raise ValueError('per_genre deve ser positivo')
    with tracks_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        columns = list(zip(next(reader), next(reader)))
        next(reader)  # linha track_id
        records = {}
        for values in reader:
            row = dict(zip(columns, values))
            track_id = int(values[0])
            if track_id in records:
                raise ValueError(f'ID repetido: {track_id}')
            records[track_id] = row
    small = {i for i, r in records.items() if r[('set', 'subset')] == 'small'}
    artists = {records[i][('artist', 'id')] for i in small}
    candidates = {
        i: r for i, r in records.items()
        if r[('set', 'subset')] == 'medium'
        and r[('track', 'genre_top')] in GENRES
        and r[('artist', 'id')] not in {'', '0'}
        and r[('track', 'title')].strip()
    }
    # Igualdade numérica exata dos 518 atributos; não prova identidade do áudio.
    signatures = {}
    wanted = small | candidates.keys()
    with features_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        for _ in range(4):
            next(reader)
        for values in reader:
            track_id = int(values[0])
            if track_id not in wanted:
                continue
            numbers = [float(v) for v in values[1:]]
            if len(numbers) != 518 or not all(map(math.isfinite, numbers)):
                continue
            numbers = [v if v else 0.0 for v in numbers]  # normaliza -0.0
            signatures[track_id] = hashlib.sha256(struct.pack('!518d', *numbers)).hexdigest()
    small_signatures = {signatures[i] for i in small if i in signatures}
    eligible = {i: r for i, r in candidates.items()
                if i in signatures and signatures[i] not in small_signatures}
    selected = []
    used_artists, used_signatures = set(), set()
    for genre in GENRES:
        ids = [i for i, r in eligible.items() if r[('track', 'genre_top')] == genre]
        ids.sort(key=lambda i: (eligible[i][('artist', 'id')] in artists,
                               hashlib.sha256(f'{seed}:{genre}:{i}'.encode()).hexdigest(), i))
        group = []
        for track_id in ids:
            row = eligible[track_id]
            artist_id = row[('artist', 'id')]
            if artist_id in used_artists or signatures[track_id] in used_signatures:
                continue
            used_artists.add(artist_id)
            used_signatures.add(signatures[track_id])
            name = f'{track_id:06d}'
            group.append(dict(track_id=track_id, genre=genre,
                              artist_id=artist_id, artist=row[('artist', 'name')],
                              artist_seen_in_small=artist_id in artists,
                              title=row[('track', 'title')],
                              license=row[('track', 'license')],
                              subset='medium', relative_path=f'{name[:3]}/{name}.mp3',
                              feature_sha256=signatures[track_id]))
            if len(group) == per_genre:
                break
        if len(group) != per_genre:
            raise ValueError(f'{genre}: somente {len(group)} faixas elegíveis; nenhuma saída criada')
        selected.extend(group)
    provenance = dict(
        seed=seed, per_genre=per_genre,
        method='genres sorted; unseen artist first; ascending SHA256(UTF-8 seed:genre:track_id), then track_id',
        exclusions=['all Small IDs', 'unknown artist ID',
                    'missing title', 'missing/nonfinite official features',
                    'exact official feature duplicate of Small',
                    'repeated selected artist or exact feature vector'],
        small_count=len(small), small_with_valid_features=len(small & signatures.keys()),
        candidates_by_genre={g: sum(r[('track', 'genre_top')] == g for r in eligible.values())
                             for g in GENRES},
        tracks_sha256=file_hash(tracks_path), features_sha256=file_hash(features_path),
        script_sha256=file_hash(Path(__file__)),
        source='https://github.com/mdeff/fma#data',
        audio_archive='fma_medium.zip',
        scope='unseen tracks within FMA; unseen artists preferred; not a separate dataset',
        status='selected; audio not downloaded or validated; no predictions produced',
        limitations=['feature equality is not audio fingerprinting',
                     'artist IDs do not identify every alias or collaboration',
                     '30-second archive excerpts need not reproduce full-track official features'])
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (output / 'mp3_paths.txt').write_text(''.join(f"{r['relative_path']}\n" for r in selected), encoding='utf-8')
    provenance['manifest_sha256'] = file_hash(output / 'manifest.csv')
    (output / 'selection.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in selected:
        print(f"{row['genre']}: {row['relative_path']} | {row['artist']} — {row['title']}")
    print(f'Seleção: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, default=ROOT / 'data/fma_metadata/tracks.csv')
    parser.add_argument('--features', type=Path, default=ROOT / 'data/fma_metadata/features.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/unseen_fma/sample')
    parser.add_argument('--per-genre', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    select(args.tracks, args.features, args.output, args.per_genre, args.seed)


if __name__ == '__main__':
    main()
