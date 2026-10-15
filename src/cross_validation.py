from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


LOGGER = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]

TIME_COLUMNS = [
    "training_time_seconds",
    "prediction_time_seconds",
]


@dataclass(slots=True)
class CrossValidationResults:
    """
    Resultados produzidos pela validação cruzada.

    fold_results:
        Métricas obtidas por classificador em cada fold.

    metrics_summary:
        Média e desvio padrão das métricas e dos tempos por modelo.

    predictions:
        Predições out-of-fold de todas as amostras para cada modelo.

    fold_assignments:
        Fold de teste atribuído a cada faixa.
    """

    fold_results: pd.DataFrame
    metrics_summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    track_ids: pd.Series,
    models: Mapping[str, BaseEstimator],
    *,
    n_splits: int = 10,
    random_state: int = 42,
) -> CrossValidationResults:
    """
    Executa validação cruzada estratificada sobre todos os modelos.

    As mesmas divisões de treino e teste são utilizadas por todos os
    classificadores. Cada pipeline é clonado antes do treinamento,
    garantindo que nenhum estado ajustado seja compartilhado entre
    folds ou modelos.

    Parameters
    ----------
    X:
        Matriz contendo apenas os atributos numéricos.

    y:
        Rótulos dos gêneros musicais.

    track_ids:
        Identificadores das faixas, usados para rastreabilidade.

    models:
        Mapeamento entre nome do modelo e estimador ou pipeline.

    n_splits:
        Quantidade de folds da validação cruzada.

    random_state:
        Semente usada no embaralhamento estratificado.

    Returns
    -------
    CrossValidationResults
        Métricas, predições e atribuições dos folds.
    """

    _validate_inputs(
        X=X,
        y=y,
        track_ids=track_ids,
        models=models,
        n_splits=n_splits,
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Materializa os índices uma única vez.
    # Todos os classificadores receberão exatamente os mesmos folds.
    splits = list(splitter.split(X, y))

    fold_assignments = _build_fold_assignments(
        y=y,
        track_ids=track_ids,
        splits=splits,
    )

    fold_records: list[dict[str, int | float | str]] = []
    prediction_records: list[dict[str, int | str]] = []

    total_training_runs = len(models) * n_splits
    completed_runs = 0

    LOGGER.info(
        "Iniciando validação cruzada: %d modelos, %d folds, %d treinamentos.",
        len(models),
        n_splits,
        total_training_runs,
    )

    for model_name, model_template in models.items():
        LOGGER.info("Avaliando modelo: %s", model_name)

        for fold_number, (train_indices, test_indices) in enumerate(
            splits,
            start=1,
        ):
            completed_runs += 1

            LOGGER.info(
                "[%d/%d] Modelo=%s | Fold=%d/%d",
                completed_runs,
                total_training_runs,
                model_name,
                fold_number,
                n_splits,
            )

            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            test_track_ids = track_ids.iloc[test_indices]

            # Cada execução recebe uma instância nova e não ajustada.
            model = clone(model_template)

            try:
                training_start = perf_counter()
                model.fit(X_train, y_train)
                training_time = perf_counter() - training_start

                prediction_start = perf_counter()
                y_pred = model.predict(X_test)
                prediction_time = perf_counter() - prediction_start

            except Exception as error:
                LOGGER.exception(
                    "Falha no modelo %s, fold %d.",
                    model_name,
                    fold_number,
                )

                raise RuntimeError(
                    f"Erro ao executar {model_name} no fold "
                    f"{fold_number}."
                ) from error

            fold_metrics = _calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
            )

            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    **fold_metrics,
                    "training_time_seconds": training_time,
                    "prediction_time_seconds": prediction_time,
                }
            )

            prediction_records.extend(
                _create_prediction_records(
                    model_name=model_name,
                    fold_number=fold_number,
                    track_ids=test_track_ids,
                    y_true=y_test,
                    y_pred=y_pred,
                )
            )

            LOGGER.info(
                "Resultado | %s | Fold %d | "
                "Accuracy=%.4f | Precision=%.4f | "
                "Recall=%.4f | F1=%.4f | "
                "Treino=%.2fs | Predição=%.2fs",
                model_name,
                fold_number,
                fold_metrics["accuracy"],
                fold_metrics["precision_macro"],
                fold_metrics["recall_macro"],
                fold_metrics["f1_macro"],
                training_time,
                prediction_time,
            )

    fold_results = pd.DataFrame(fold_records)
    predictions = pd.DataFrame(prediction_records)

    _validate_output_completeness(
        predictions=predictions,
        sample_count=len(X),
        model_count=len(models),
    )

    metrics_summary = summarize_metrics(fold_results)

    LOGGER.info("Validação cruzada concluída.")
    LOGGER.info(
        "Ranking por F1-score macro:\n%s",
        metrics_summary[
            [
                "rank",
                "model",
                "f1_macro_mean",
                "f1_macro_std",
            ]
        ].to_string(index=False),
    )

    return CrossValidationResults(
        fold_results=fold_results,
        metrics_summary=metrics_summary,
        predictions=predictions,
        fold_assignments=fold_assignments,
    )


def save_cross_validation_results(
    results: CrossValidationResults,
    output_directory: str | Path,
) -> None:
    """
    Salva em CSV os resultados produzidos pela validação cruzada.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    results.fold_results.to_csv(
        output_directory / "fold_results.csv",
        index=False,
        float_format="%.10f",
    )

    results.metrics_summary.to_csv(
        output_directory / "metrics_summary.csv",
        index=False,
        float_format="%.10f",
    )

    results.predictions.to_csv(
        output_directory / "predictions.csv",
        index=False,
    )

    results.fold_assignments.to_csv(
        output_directory / "fold_assignments.csv",
        index=False,
    )

    LOGGER.info(
        "Resultados da validação cruzada salvos em: %s",
        output_directory.resolve(),
    )


def _calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula as métricas definidas no protocolo experimental.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def _create_prediction_records(
    *,
    model_name: str,
    fold_number: int,
    track_ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[dict[str, int | str]]:
    """
    Constrói registros individuais das predições out-of-fold.
    """

    records: list[dict[str, int | str]] = []

    for track_id, true_label, predicted_label in zip(
        track_ids.to_numpy(),
        y_true.to_numpy(),
        y_pred,
        strict=True,
    ):
        records.append(
            {
                "model": model_name,
                "fold": fold_number,
                "track_id": track_id,
                "true_label": str(true_label),
                "predicted_label": str(predicted_label),
            }
        )

    return records


def _build_fold_assignments(
    *,
    y: pd.Series,
    track_ids: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Registra em qual fold de teste cada amostra foi posicionada.
    """

    assigned_folds = np.full(
        shape=len(y),
        fill_value=-1,
        dtype=np.int16,
    )

    for fold_number, (_, test_indices) in enumerate(
        splits,
        start=1,
    ):
        if np.any(assigned_folds[test_indices] != -1):
            raise RuntimeError(
                "Uma amostra foi atribuída a mais de um fold de teste."
            )

        assigned_folds[test_indices] = fold_number

    if np.any(assigned_folds == -1):
        raise RuntimeError(
            "Há amostras que não foram atribuídas a nenhum fold."
        )

    return pd.DataFrame(
        {
            "track_id": track_ids.to_numpy(),
            "genre": y.to_numpy(),
            "test_fold": assigned_folds,
        }
    ).sort_values("track_id").reset_index(drop=True)


def summarize_metrics(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida média, desvio padrão, mediana, mínimo 
    e máximo por classificador.
    """

    columns_to_summarize = [
        *METRIC_COLUMNS,
        *TIME_COLUMNS,
    ]

    summary = (
        fold_results
        .groupby("model", sort=False)[columns_to_summarize]
        .agg(["mean", "std", "median", "min", "max"])
    )

    summary.columns = [
        f"{column}_{statistic}"
        for column, statistic in summary.columns
    ]

    summary = summary.reset_index()

    summary = (
        summary
        .sort_values(
            by="f1_macro_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.insert(
        loc=0,
        column="rank",
        value=np.arange(1, len(summary) + 1),
    )

    return summary


def _validate_inputs(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    track_ids: pd.Series,
    models: Mapping[str, BaseEstimator],
    n_splits: int,
) -> None:
    """
    Verifica condições necessárias antes da execução.
    """

    if X.empty:
        raise ValueError("A matriz de atributos está vazia.")

    if len(X) != len(y) or len(X) != len(track_ids):
        raise ValueError(
            "X, y e track_ids possuem quantidades diferentes "
            "de registros."
        )

    if y.isna().any():
        raise ValueError("Existem rótulos ausentes em y.")

    if track_ids.isna().any():
        raise ValueError("Existem track_ids ausentes.")

    if track_ids.duplicated().any():
        raise ValueError("Existem track_ids duplicados.")

    if not models:
        raise ValueError("Nenhum modelo foi fornecido.")

    if n_splits < 2:
        raise ValueError("n_splits deve ser igual ou superior a 2.")

    smallest_class = int(y.value_counts().min())

    if n_splits > smallest_class:
        raise ValueError(
            f"n_splits={n_splits} é maior que a menor classe, "
            f"que possui {smallest_class} observações."
        )


def _validate_output_completeness(
    *,
    predictions: pd.DataFrame,
    sample_count: int,
    model_count: int,
) -> None:
    """
    Confirma que cada modelo produziu uma predição out-of-fold
    para cada amostra.
    """

    expected_prediction_count = sample_count * model_count

    if len(predictions) != expected_prediction_count:
        raise RuntimeError(
            "Quantidade de predições incompatível. "
            f"Esperado={expected_prediction_count}, "
            f"obtido={len(predictions)}."
        )

    predictions_per_model = predictions.groupby("model").size()

    if not (predictions_per_model == sample_count).all():
        raise RuntimeError(
            "Um ou mais modelos não geraram uma predição "
            "para cada amostra."
        )

    duplicate_predictions = predictions.duplicated(
        subset=["model", "track_id"]
    ).sum()

    if duplicate_predictions:
        raise RuntimeError(
            f"Foram encontradas {duplicate_predictions} "
            "predições repetidas por modelo e track_id."
        )
