from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

ID_COLUMN = "track_id"
TARGET_COLUMN = "genre"
EXPECTED_FEATURE_COUNT = 518

EXPECTED_GENRES = {
    "Hip-Hop",
    "Pop",
    "Folk",
    "Experimental",
    "Rock",
    "International",
    "Electronic",
    "Instrumental",
}


@dataclass(slots=True)
class DatabaseBundle:
    """
    Estrutura contendo os dados já carregados e validados.

    X:
        Matriz de atributos, sem track_id e sem gênero.
    y:
        Rótulos dos gêneros musicais.
    track_ids:
        Identificadores das faixas, preservados para rastreabilidade.
    class_names:
        Classes encontradas no dataset, em ordem alfabética.
    diagnostics:
        Informações gerais sobre o conjunto carregado.
    duplicate_samples:
        Registros envolvidos em grupos de duplicatas exatas.
    """

    X: pd.DataFrame
    y: pd.Series
    track_ids: pd.Series
    class_names: tuple[str, ...]
    diagnostics: dict[str, int | str]
    duplicate_samples: pd.DataFrame


def load_dataset(
    csv_path: str | Path,
    *,
    drop_duplicate_samples: bool = False,
    validate_expected_structure: bool = True,
) -> DatabaseBundle:
    """
    Carrega e valida o dataset FMA Small previamente preparado.

    Parameters
    ----------
    csv_path:
        Caminho para o arquivo CSV.

    drop_duplicate_samples:
        Quando True, remove registros com os mesmos 518 atributos e
        o mesmo gênero, mantendo a primeira ocorrência.

        O track_id não participa dessa verificação, pois identificadores
        distintos podem corresponder a observações acusticamente idênticas.

    validate_expected_structure:
        Quando True, exige exatamente 518 atributos e os oito gêneros
        previstos para a FMA Small.

    Returns
    -------
    DatasetBundle
        Dados separados e informações de diagnóstico.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    ValueError
        Se o dataset tiver estrutura inválida, nulos, infinitos,
        identificadores repetidos ou classes inesperadas.

    TypeError
        Se alguma feature não for numérica.
    """

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {csv_path.resolve()}")

    LOGGER.info("Carregado dataset: %s", csv_path)
    dataframe = pd.read_csv(csv_path)

    _validate_required_columns(dataframe)

    # Remove espaços acidentais nos nomes das classes
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype("string").str.strip()

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]

    if validate_expected_structure:
        _validate_expected_structure(dataframe, feature_columns)

    _validate_identifiers(dataframe)
    _validate_numeric_features(dataframe, feature_columns)
    _validate_missing_and_infinite_values(dataframe, feature_columns)

    duplicate_samples = _find_duplicate_samples(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    original_sample_count = len(dataframe)

    if not duplicate_samples.empty:
        redundant_count = int((duplicate_samples["decision"] == "remove").sum())

        LOGGER.warning(
            "Foram encontradas %d observações redundantes em %d grupos.",
            redundant_count,
            duplicate_samples["duplicate_group"].nunique(),
        )

        if drop_duplicate_samples:
            duplicate_subset = [*feature_columns, TARGET_COLUMN]

            dataframe = dataframe.drop_duplicates(
                subset=duplicate_subset,
                keep="first",
            ).reset_index(drop=True)

            LOGGER.info(
                "%d observações redundantes foram removidas.",
                redundant_count,
            )

    X = dataframe.loc[:, feature_columns].copy()
    y = dataframe.loc[:, TARGET_COLUMN].copy()
    track_ids = dataframe.loc[:, ID_COLUMN].copy()

    constant_features = X.loc[:, X.nunique(dropna=False) <= 1].columns.tolist()

    if constant_features:
        LOGGER.warning(
            "Foram encontradas %d features constantes.",
            len(constant_features),
        )

    class_names = tuple(sorted(y.unique().tolist()))
    class_distribution = y.value_counts().sort_index()

    diagnostics: dict[str, int | str] = {
        "source_file": csv_path.name,
        "original_samples": original_sample_count,
        "final_samples": len(X),
        "number_of_features": X.shape[1],
        "number_of_classes": y.nunique(),
        "missing_values": int(X.isna().sum().sum()),
        "infinite_values": int(
            np.isinf(X.to_numpy(dtype=np.float64, copy=False)).sum()
        ),
        "duplicate_track_ids": int(track_ids.duplicated().sum()),
        "duplicate_groups": int(
            duplicate_samples["duplicate_group"].nunique()
            if not duplicate_samples.empty
            else 0
        ),
        "redundant_samples": int(
            (duplicate_samples["decision"] == "remove").sum()
            if not duplicate_samples.empty
            else 0
        ),
        "constant_features": len(constant_features),
    }

    LOGGER.info(
        "Dataset pronto: X=%s, y=%s, classes=%d.",
        X.shape,
        y.shape,
        y.nunique(),
    )

    LOGGER.info(
        "Distribuição das classes: \n%s",
        class_distribution.to_string(),
    )

    return DatabaseBundle(
        X=X,
        y=y,
        track_ids=track_ids,
        class_names=class_names,
        diagnostics=diagnostics,
        duplicate_samples=duplicate_samples,
    )


def save_dataset_reports(
    bundle: DatabaseBundle,
    output_directory: str | Path,
) -> None:
    """
    Salva relatórios básicos de qualidade e caracterização do dataset.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "metric": list(bundle.diagnostics.keys()),
            "value": list(bundle.diagnostics.values()),
        }
    )

    class_distribution = (
        bundle.y.value_counts()
        .sort_index()
        .rename_axis("genre")
        .reset_index(name="sample_count")
    )

    feature_summary = bundle.X.describe().transpose().reset_index(names="feature")

    summary.to_csv(
        output_directory / "dataset_summary.csv",
        index=False,
    )

    class_distribution.to_csv(
        output_directory / "class_distribution.csv",
        index=False,
    )

    feature_summary.to_csv(
        output_directory / "feature_summary.csv",
        index=False,
    )

    if not bundle.duplicate_samples.empty:
        bundle.duplicate_samples.to_csv(
            output_directory / "duplicate_samples.csv",
            index=False,
        )

    LOGGER.info(
        "Relatórios do dataset salvos em: %s",
        output_directory.resolve(),
    )


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: " f"{sorted(missing_columns)}"
        )


def _validate_expected_structure(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Esperadas {EXPECTED_FEATURE_COUNT} features,"
            f"mas foram encontradas {len(feature_columns)}."
        )

    genres = set(dataframe[TARGET_COLUMN].dropna().unique())

    missing_genres = EXPECTED_GENRES.difference(genres)
    unexpected_genres = genres.difference(EXPECTED_GENRES)

    if missing_genres or unexpected_genres:
        raise ValueError(
            "Classes incompatíveis com a FMA Small. "
            f"Ausentes: {sorted(missing_genres)}. "
            f"Inesperadas: {sorted(unexpected_genres)}."
        )

    class_counts = dataframe[TARGET_COLUMN].value_counts()

    if not (class_counts == 1_000).all():
        raise ValueError(
            "A distribuição original não possui 1.000 registros "
            f"por gênero:\n{class_counts.to_string()}"
        )


def _validate_identifiers(dataframe: pd.DataFrame) -> None:
    if dataframe[ID_COLUMN].isna().any():
        raise ValueError("Há valores ausentes na coluna track_id.")

    duplicate_track_ids = dataframe[ID_COLUMN].duplicated().sum()

    if duplicate_track_ids:
        raise ValueError(
            f"Foram encontrados {duplicate_track_ids} tracks_ids repetidos."
        )


def _validate_numeric_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if non_numeric_features:
        preview = non_numeric_features[:10]

        raise TypeError(
            "Foram encontradas features não numéricas. "
            f"Primeiras ocorrências: {preview}"
        )


def _validate_missing_and_infinite_values(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    features = dataframe.loc[:, feature_columns]

    missing_count = int(features.isna().sum().sum())

    if missing_count:
        raise ValueError(f"Foram encontrados {missing_count} valores ausentes.")

    feature_array = features.to_numpy(
        dtype=np.float64,
        copy=False,
    )

    infinite_count = int(np.isinf(feature_array).sum())

    if infinite_count:
        raise ValueError(f"Foram encontrados {infinite_count} valores infinitos.")


def _find_duplicate_samples(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Localiza registros com os mesmos atributos e o mesmo gênero;

    O track_id é desconsiderado na definição de duplicata.
    """

    duplicate_subset = [*feature_columns, TARGET_COLUMN]

    duplicate_mask = dataframe.duplicated(subset=duplicate_subset, keep=False)

    if not duplicate_mask.any():
        return pd.DataFrame(
            columns=[
                ID_COLUMN,
                TARGET_COLUMN,
                "duplicate_group",
                "decision",
            ]
        )

    duplicate_rows = dataframe.loc[
        duplicate_mask,
        [ID_COLUMN, TARGET_COLUMN, *feature_columns],
    ].copy()

    row_hashes = pd.util.hash_pandas_object(
        duplicate_rows[duplicate_subset],
        index=False,
    )

    duplicate_rows["duplicate_group"] = pd.factorize(row_hashes)[0] + 1

    duplicate_rows["decision"] = np.where(
        duplicate_rows.groupby("duplicate_group").cumcount() == 0,
        "keep",
        "remove",
    )

    return (
        duplicate_rows[
            [
                ID_COLUMN,
                TARGET_COLUMN,
                "duplicate_group",
                "decision",
            ]
        ]
        .sort_values(["duplicate_group", ID_COLUMN])
        .reset_index(drop=True)
    )
