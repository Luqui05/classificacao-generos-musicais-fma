"""Diagnóstico numérico do extrator contra features.csv, sem treinar modelos."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd

from src.audio_features import FEATURE_NAMES, extract_features, validate_features
from src.data_loader import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata
KNOWN_SHORT_SMALL = {98565, 98567, 98569, 99134, 108925, 133297}
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def audio_relative_path(track_id: int) -> str:
    if not 1 <= track_id <= 999999:
        raise ValueError(f"track_id inválido: {track_id}")
    name = f"{track_id:06d}"
    return f"{name[:3]}/{name}.mp3"


def read_official_features(path: Path, track_ids: list[int]) -> pd.DataFrame:
    """Lê o cabeçalho de três níveis e retém apenas as linhas solicitadas."""
    requested = set(track_ids)
    if not requested or len(requested) != len(track_ids):
        raise ValueError("Informe track_ids únicos e não vazios.")
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = [next(reader) for _ in range(3)]
        columns = tuple("_".join(parts) for parts in zip(*(h[1:] for h in headers), strict=True))
        if columns != FEATURE_NAMES:
            raise ValueError("features.csv oficial não possui o esquema esperado de 518 colunas.")
        index_header = next(reader)
        if index_header[0] != "track_id" or any(index_header[1:]):
            raise ValueError("Cabeçalho track_id inválido no features.csv oficial.")
        for row in reader:
            track_id = int(row[0])
            if track_id not in requested:
                continue
            if track_id in rows:
                raise ValueError(f"track_id duplicado na referência oficial: {track_id}")
            values = np.asarray(row[1:], dtype=np.float64)
            if values.shape != (518,) or not np.isfinite(values).all():
                raise ValueError(f"Referência inválida ou não finita para track_id={track_id}.")
            rows[track_id] = values
    missing = requested.difference(rows)
    if missing:
        raise ValueError(f"Faixas ausentes da referência oficial: {sorted(missing)}")
    return pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_NAMES).loc[track_ids]


def prepare_sample(dataset_path: Path, official_path: Path, output: Path,
                   *, per_genre: int = 2, seed: int = 42) -> pd.DataFrame:
    if per_genre < 1:
        raise ValueError("per_genre deve ser positivo.")
    if output.exists():
        raise FileExistsError(f"A pasta de saída já existe: {output}")
    dataset = load_dataset(dataset_path, drop_duplicate_samples=True)
    candidates = pd.DataFrame({"track_id": dataset.track_ids, "genre": dataset.y})
    candidates = candidates[~candidates.track_id.isin(KNOWN_SHORT_SMALL)]
    rng = np.random.default_rng(seed)
    selected = []
    for genre, group in candidates.groupby("genre", sort=True):
        group = group.sort_values("track_id")
        if len(group) < per_genre:
            raise ValueError(f"Poucas faixas para o gênero {genre}.")
        positions = rng.choice(len(group), size=per_genre, replace=False)
        selected.append(group.iloc[positions].sort_values("track_id"))
    manifest = pd.concat(selected, ignore_index=True)
    manifest["relative_path"] = manifest.track_id.map(audio_relative_path)
    reference = read_official_features(official_path, manifest.track_id.tolist())
    local = dataset.X.set_axis(dataset.track_ids).loc[reference.index, list(FEATURE_NAMES)]
    # Permite apenas arredondamento na leitura/exportação dos CSVs, sem usar o
    # CSV derivado como substituto dos valores oficiais na comparação de áudio.
    if not np.allclose(local.to_numpy(), reference.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("As linhas selecionadas do CSV do experimento divergem da referência oficial.")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "sorted genres and track_ids; numpy.default_rng.choice without replacement",
        "seed": seed, "per_genre": per_genre,
        "excluded_short_track_ids": sorted(KNOWN_SHORT_SMALL),
        "redundancies_removed": dataset.diagnostics["redundant_samples"],
        "dataset_sha256": sha256(dataset_path),
        "official_features_sha256": sha256(official_path),
        "numpy_version": np.__version__,
        "reference_source": "https://github.com/mdeff/fma#data",
        "audio_source_required": "unmodified MP3 excerpts from fma_small.zip (rc1)",
        "status": "awaiting_official_audio; numerical_equivalence_not_established",
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    provenance["manifest_sha256"] = sha256(output / "manifest.csv")
    write_json(output / "selection.json", provenance)
    return manifest


def compare_vectors(reference: pd.Series, extracted: pd.Series,
                    *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> pd.DataFrame:
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("As tolerâncias devem ser finitas e não negativas.")
    if tuple(reference.index) != FEATURE_NAMES:
        raise ValueError("Esquema da referência inesperado.")
    expected = reference.to_numpy(dtype=np.float64)
    if not np.isfinite(expected).all():
        raise ValueError("Referência contém valores não finitos.")
    actual = validate_features(extracted).to_numpy(dtype=np.float64)
    absolute = np.abs(actual - expected)
    # Erro relativo indefinido quando a referência é zero; CSV contém campo vazio.
    relative = np.full(518, np.nan)
    np.divide(absolute, np.abs(expected), out=relative, where=expected != 0)
    parts = [name.rsplit("_", 2) for name in FEATURE_NAMES]
    return pd.DataFrame({
        "feature": FEATURE_NAMES, "family": [p[0] for p in parts],
        "statistic": [p[1] for p in parts], "number": [p[2] for p in parts],
        "official": expected, "extracted": actual, "absolute_error": absolute,
        "relative_error": relative, "reference_is_zero": expected == 0,
        "within_tolerance": absolute <= atol + rtol * np.abs(expected),
    })


def summarize_differences(details: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return details.groupby(groups, sort=True).agg(
        comparisons=("within_tolerance", "size"),
        within_tolerance=("within_tolerance", "sum"),
        mean_absolute_error=("absolute_error", "mean"),
        max_absolute_error=("absolute_error", "max"),
        max_relative_error=("relative_error", "max"),
        zero_references=("reference_is_zero", "sum"),
    ).reset_index()


def inspect_audio(path: Path) -> dict:
    """Metadados do SoundFile; não afirma qual backend Librosa usará."""
    try:
        import soundfile
        info = soundfile.info(path)
        return {"inspection_backend": "soundfile", "samplerate": info.samplerate,
                "channels": info.channels, "frames": info.frames,
                "duration_seconds": info.duration, "format": info.format,
                "subtype": info.subtype, "libsndfile": soundfile.__libsndfile_version__}
    except Exception as error:
        return {"inspection_error": f"{type(error).__name__}: {error}"}


def run_comparison(sample: Path, official_path: Path, audio_root: Path, output: Path,
                   *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> int:
    """Retorna 0 para coincidência, 1 para divergência, 2 para execução incompleta."""
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("As tolerâncias devem ser finitas e não negativas.")
    if output.exists():
        raise FileExistsError(f"A pasta de saída já existe: {output}")
    selection = json.loads((sample / "selection.json").read_text(encoding="utf-8"))
    if sha256(sample / "manifest.csv") != selection["manifest_sha256"]:
        raise ValueError("Manifesto alterado desde a seleção. Gere uma nova amostra.")
    official_hash = sha256(official_path)
    if official_hash != selection["official_features_sha256"]:
        raise ValueError("features.csv alterado desde a seleção. Gere uma nova amostra.")
    manifest = pd.read_csv(sample / "manifest.csv")
    if manifest.empty or manifest.track_id.duplicated().any():
        raise ValueError("O manifesto precisa conter faixas únicas e não pode ser vazio.")
    for row in manifest.itertuples(index=False):
        if row.relative_path != audio_relative_path(int(row.track_id)):
            raise ValueError(f"Caminho incompatível com track_id={row.track_id}.")
    reference = read_official_features(official_path, manifest.track_id.tolist())
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
        "manifest_sha256": selection["manifest_sha256"],
        "official_features_sha256": official_hash,
        "extractor_sha256": sha256(PROJECT_ROOT / "src/audio_features.py"),
        "comparison_sha256": sha256(Path(__file__)),
        "audio_root": str(audio_root.resolve()), "rtol": rtol, "atol": atol,
        "criterion": "absolute_error <= atol + rtol * abs(official)",
        "tolerances_status": "exploratory; not a validated equivalence threshold",
        "librosa_decoder_backend": "automatic; not observed by this comparator",
        "audio_origin": "user supplied; archive authenticity not independently verified",
        "audio": [],
    }
    details, tracks, vectors = [], [], []
    for row in manifest.itertuples(index=False):
        path = audio_root / row.relative_path
        record = {"track_id": row.track_id, "genre": row.genre,
                  "relative_path": row.relative_path, "status": "missing",
                  "comparisons": 0, "within_tolerance": 0, "error": ""}
        probe = {"track_id": row.track_id, "path": str(path.resolve()), "warnings": []}
        if path.is_file():
            captured = []
            try:
                probe["sha256"] = sha256(path)
                probe["file_bytes"] = path.stat().st_size
                probe["info"] = inspect_audio(path)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    vector = extract_features(path)
                comparison = compare_vectors(reference.loc[row.track_id], vector, rtol=rtol, atol=atol)
                comparison.insert(0, "genre", row.genre)
                comparison.insert(0, "track_id", row.track_id)
                details.append(comparison)
                vectors.append({"track_id": row.track_id, **vector.to_dict()})
                matches = int(comparison.within_tolerance.sum())
                record.update(status="within_tolerance" if matches == 518 else "divergent",
                              comparisons=518, within_tolerance=matches)
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
            finally:
                probe["warnings"] = [f"{w.category.__name__}: {w.message}" for w in captured]
        tracks.append(record)
        metadata["audio"].append(probe)
    tracks_frame = pd.DataFrame(tracks)
    counts = tracks_frame.status.value_counts().to_dict()
    incomplete = any(status in counts for status in ["missing", "error"])
    exit_code = 2 if incomplete else (1 if "divergent" in counts else 0)
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    requested_tracks=len(manifest), status_counts=counts, exit_code=exit_code,
                    conclusion="incomplete" if incomplete else "diagnostic_only")
    output.mkdir(parents=True, exist_ok=False)
    tracks_frame.to_csv(output / "tracks.csv", index=False)
    pd.DataFrame(vectors, columns=["track_id", *FEATURE_NAMES]).to_csv(
        output / "extracted_features.csv", index=False, float_format="%.17g")
    if details:
        combined = pd.concat(details, ignore_index=True)
        combined.to_csv(output / "differences.csv", index=False, float_format="%.17g")
        for name, groups in [("family_summary", ["family"]),
                             ("family_statistic_summary", ["family", "statistic"])]:
            summarize_differences(combined, groups).to_csv(output / f"{name}.csv", index=False,
                                                         float_format="%.17g")
    write_json(output / "run.json", metadata)
    return exit_code
