from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

LOGGER = logging.getLogger(__name__)

METRICS = {
    "accuracy": "Acurácia",
    "precision_macro": "Precisão macro",
    "recall_macro": "Revocação macro",
    "f1_macro": "F1-score macro",
}

DISPLAY_MODEL_NAMES = {
    "Decision Tree": "Árvore de Decisão",
    "Naive Bayes": "Naive Bayes gaussiano",
}


def generate_visualizations(
    *,
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    average_ranks: pd.DataFrame,
    figures_directory: str | Path,
    tables_directory: str | Path,
) -> pd.DataFrame:
    """
    Gera as visualizações e tabelas auxiliares do experimento.

    Os resultados são lidos dos arquivos já produzidos pela
    validação cruzada e pela análise estatística. Nenhum modelo
    é treinado novamente.
    """

    _validate_inputs(
        fold_results=fold_results,
        predictions=predictions,
        average_ranks=average_ranks,
    )

    figures_directory = Path(figures_directory)
    tables_directory = Path(tables_directory)

    figures_directory.mkdir(parents=True, exist_ok=True)
    tables_directory.mkdir(parents=True, exist_ok=True)

    model_order = (
        fold_results.groupby("model")["f1_macro"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    class_names = sorted(predictions["true_label"].unique().tolist())

    class_metrics = build_class_metrics(
        predictions=predictions,
        model_order=model_order,
        class_names=class_names,
    )

    class_metrics.to_csv(
        tables_directory / "class_metrics.csv",
        index=False,
        float_format="%.10f",
        encoding="utf-8-sig",
    )

    plot_metrics_comparison(
        fold_results=fold_results,
        model_order=model_order,
        output_path=(figures_directory / "metrics_comparison.png"),
    )

    plot_f1_boxplot(
        fold_results=fold_results,
        model_order=model_order,
        output_path=figures_directory / "f1_boxplot.png",
    )

    plot_average_ranks(
        average_ranks=average_ranks,
        output_path=figures_directory / "average_ranks.png",
    )

    plot_class_f1_heatmap(
        class_metrics=class_metrics,
        model_order=model_order,
        class_names=class_names,
        output_path=figures_directory / "class_f1_heatmap.png",
    )

    plot_confusion_matrices(
        predictions=predictions,
        model_order=model_order,
        class_names=class_names,
        figures_directory=figures_directory,
        tables_directory=tables_directory,
    )

    LOGGER.info(
        "Visualizações salvas em: %s",
        figures_directory.resolve(),
    )

    LOGGER.info(
        "Tabelas auxiliares salvas em: %s",
        tables_directory.resolve(),
    )

    return class_metrics


def plot_metrics_comparison(
    *,
    fold_results: pd.DataFrame,
    model_order: list[str],
    output_path: str | Path,
) -> None:
    """
    Compara as médias das quatro métricas.

    As barras de erro representam um desvio padrão entre os folds.
    """

    metric_columns = list(METRICS)

    means = fold_results.groupby("model")[metric_columns].mean().reindex(model_order)

    standard_deviations = (
        fold_results.groupby("model")[metric_columns].std().reindex(model_order)
    )

    x_positions = np.arange(len(model_order))
    bar_width = 0.19

    figure, axis = plt.subplots(figsize=(12, 7))

    for metric_index, (metric, label) in enumerate(METRICS.items()):
        offset = (metric_index - 1.5) * bar_width

        axis.bar(
            x_positions + offset,
            means[metric].to_numpy(),
            width=bar_width,
            yerr=standard_deviations[metric].to_numpy(),
            capsize=3,
            label=label,
        )

    axis.set_title("Desempenho médio dos classificadores")
    axis.set_xlabel("Classificador")
    axis.set_ylabel("Valor da métrica")
    axis.set_ylim(0.0, 1.0)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        model_order,
        rotation=15,
        ha="right",
    )

    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    _save_figure(
        figure=figure,
        output_path=output_path,
    )


def plot_f1_boxplot(
    *,
    fold_results: pd.DataFrame,
    model_order: list[str],
    output_path: str | Path,
) -> None:
    """
    Exibe a distribuição do F1-score macro entre os folds.
    """

    values_by_model = [
        fold_results.loc[
            fold_results["model"] == model_name,
            "f1_macro",
        ].to_numpy()
        for model_name in model_order
    ]

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.boxplot(
        values_by_model,
        tick_labels=model_order,
        showmeans=True,
    )

    axis.set_title("Distribuição do F1-score macro entre os folds")
    axis.set_xlabel("Classificador")
    axis.set_ylabel("F1-score macro")
    axis.set_ylim(0.0, 1.0)

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    axis.grid(axis="y", alpha=0.25)

    _save_figure(
        figure=figure,
        output_path=output_path,
    )


def plot_average_ranks(
    *,
    average_ranks: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Exibe os ranks médios dos classificadores.

    Rank menor representa melhor colocação.
    """

    ordered_ranks = average_ranks.sort_values(
        by=["average_rank", "f1_macro_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    y_positions = np.arange(len(ordered_ranks))

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.scatter(
        ordered_ranks["average_rank"],
        y_positions,
        s=80,
    )

    for row_index, row in ordered_ranks.iterrows():
        axis.annotate(
            f"{row['average_rank']:.2f}",
            (
                row["average_rank"],
                row_index,
            ),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )

    axis.set_title("Rank médio dos classificadores")
    axis.set_xlabel("Rank médio — menor valor indica melhor posição")
    axis.set_ylabel("Classificador")

    axis.set_yticks(y_positions)
    axis.set_yticklabels(ordered_ranks["model"])

    axis.set_xticks(np.arange(1, len(ordered_ranks) + 1))

    axis.set_xlim(
        0.75,
        len(ordered_ranks) + 0.25,
    )

    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)

    _save_figure(
        figure=figure,
        output_path=output_path,
    )


def build_class_metrics(
    *,
    predictions: pd.DataFrame,
    model_order: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """
    Calcula precisão, revocação e F1-score para cada gênero.

    São utilizadas as predições out-of-fold produzidas durante
    a validação cruzada.
    """

    records: list[dict[str, int | float | str]] = []

    for model_name in model_order:
        model_predictions = predictions.loc[predictions["model"] == model_name]

        precision, recall, f1_scores, supports = precision_recall_fscore_support(
            model_predictions["true_label"],
            model_predictions["predicted_label"],
            labels=class_names,
            zero_division=0,
        )

        for (
            class_name,
            class_precision,
            class_recall,
            class_f1,
            class_support,
        ) in zip(
            class_names,
            precision,
            recall,
            f1_scores,
            supports,
            strict=True,
        ):
            records.append(
                {
                    "model": model_name,
                    "class_name": class_name,
                    "precision": float(class_precision),
                    "recall": float(class_recall),
                    "f1_score": float(class_f1),
                    "support": int(class_support),
                }
            )

    return pd.DataFrame(records)


def plot_class_f1_heatmap(
    *,
    class_metrics: pd.DataFrame,
    model_order: list[str],
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """
    Compara o F1-score de cada gênero entre os classificadores.
    """

    matrix = class_metrics.pivot(
        index="model",
        columns="class_name",
        values="f1_score",
    ).reindex(
        index=model_order,
        columns=class_names,
    )

    figure, axis = plt.subplots(figsize=(11, 5.5))

    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        label="F1 por gênero",
    )

    colorbar.ax.tick_params(labelsize=12)
    colorbar.ax.yaxis.label.set_size(13)

    axis.set_title("F1-score por gênero musical e classificador")
    axis.set_xlabel("Gênero musical")
    axis.set_ylabel("Classificador")

    axis.set_xticks(np.arange(len(class_names)))
    axis.set_xticklabels(
        class_names,
        rotation=35,
        ha="right",
    )

    axis.set_yticks(np.arange(len(model_order)))
    axis.set_yticklabels(
        [DISPLAY_MODEL_NAMES.get(model_name, model_name) for model_name in model_order]
    )

    axis.tick_params(axis="both", labelsize=13)
    axis.xaxis.label.set_size(14)
    axis.yaxis.label.set_size(14)

    axis.set_title("")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color=(
                    "white" if matrix.iloc[row_index, column_index] < 0.5 else "black"
                ),
            )

    _save_figure(
        figure=figure,
        output_path=output_path,
    )


def plot_confusion_matrices(
    *,
    predictions: pd.DataFrame,
    model_order: list[str],
    class_names: list[str],
    figures_directory: str | Path,
    tables_directory: str | Path,
) -> None:
    """
    Gera uma matriz de confusão normalizada por classificador.

    Também salva em CSV as matrizes absoluta e normalizada.
    """

    figures_directory = Path(figures_directory)
    tables_directory = Path(tables_directory)

    for model_name in model_order:
        model_predictions = predictions.loc[predictions["model"] == model_name]

        absolute_matrix = confusion_matrix(
            model_predictions["true_label"],
            model_predictions["predicted_label"],
            labels=class_names,
        )

        normalized_matrix = confusion_matrix(
            model_predictions["true_label"],
            model_predictions["predicted_label"],
            labels=class_names,
            normalize="true",
        )

        absolute_dataframe = pd.DataFrame(
            absolute_matrix,
            index=class_names,
            columns=class_names,
        )

        normalized_dataframe = pd.DataFrame(
            normalized_matrix,
            index=class_names,
            columns=class_names,
        )

        model_slug = _slugify_model_name(model_name)

        absolute_dataframe.to_csv(
            tables_directory / f"confusion_matrix_{model_slug}_absolute.csv",
            index_label="true_label",
            encoding="utf-8-sig",
        )

        normalized_dataframe.to_csv(
            tables_directory / f"confusion_matrix_{model_slug}_normalized.csv",
            index_label="true_label",
            float_format="%.10f",
            encoding="utf-8-sig",
        )

        _plot_single_confusion_matrix(
            matrix=normalized_dataframe,
            model_name=model_name,
            output_path=(figures_directory / f"confusion_matrix_{model_slug}.png"),
        )


def _plot_single_confusion_matrix(
    *,
    matrix: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
) -> None:
    """
    Plota uma matriz normalizada pela classe verdadeira.
    """

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        label="Proporção por classe verdadeira",
    )
    colorbar.ax.tick_params(labelsize=12)
    colorbar.ax.yaxis.label.set_size(13)

    axis.set_title(f"Matriz de confusão normalizada — {model_name}")
    axis.set_xlabel("Classe predita")
    axis.set_ylabel("Classe verdadeira")

    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(
        matrix.columns,
        rotation=40,
        ha="right",
    )

    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)

    axis.tick_params(axis="both", labelsize=13)
    axis.xaxis.label.set_size(14)
    axis.yaxis.label.set_size(14)
    axis.set_title("")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color=(
                    "white" if matrix.iloc[row_index, column_index] < 0.5 else "black"
                ),
            )

    _save_figure(
        figure=figure,
        output_path=output_path,
    )


def _validate_inputs(
    *,
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    average_ranks: pd.DataFrame,
) -> None:
    """
    Valida os arquivos utilizados pelas visualizações.
    """

    required_fold_columns = {
        "model",
        "fold",
        *METRICS,
    }

    required_prediction_columns = {
        "model",
        "fold",
        "track_id",
        "true_label",
        "predicted_label",
    }

    required_rank_columns = {
        "model",
        "average_rank",
        "f1_macro_mean",
    }

    datasets = [
        (
            "fold_results",
            fold_results,
            required_fold_columns,
        ),
        (
            "predictions",
            predictions,
            required_prediction_columns,
        ),
        (
            "average_ranks",
            average_ranks,
            required_rank_columns,
        ),
    ]

    for dataset_name, dataframe, required_columns in datasets:
        missing_columns = required_columns.difference(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Colunas ausentes em {dataset_name}: " f"{sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise ValueError(f"{dataset_name} está vazio.")

    fold_models = set(fold_results["model"])
    prediction_models = set(predictions["model"])
    rank_models = set(average_ranks["model"])

    if not (fold_models == prediction_models == rank_models):
        raise ValueError("Os três arquivos não contêm os mesmos modelos.")

    duplicate_predictions = predictions.duplicated(subset=["model", "track_id"]).sum()

    if duplicate_predictions:
        raise ValueError(
            f"Foram encontradas {duplicate_predictions} "
            "predições duplicadas por modelo e track_id."
        )


def _save_figure(
    *,
    figure: plt.Figure,
    output_path: str | Path,
) -> None:
    """
    Salva e fecha a figura.
    """

    output_path = Path(output_path)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    LOGGER.info(
        "Figura salva: %s",
        output_path.resolve(),
    )


def _slugify_model_name(model_name: str) -> str:
    """
    Converte o nome do modelo para um nome de arquivo.
    """

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        model_name.lower(),
    )

    return slug.strip("_")
