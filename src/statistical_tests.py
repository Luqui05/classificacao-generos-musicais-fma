from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "model",
    "fold",
    "f1_macro",
}


@dataclass(slots=True)
class StatisticalTestResults:
    """
    Resultados da análise estatística dos classificadores.

    scores_by_fold:
        F1-score macro de cada modelo em cada fold.

    ranks_by_fold:
        Posição de cada modelo em cada fold. Rank 1 representa
        o melhor desempenho.

    average_ranks:
        Rank médio e F1-score médio de cada classificador.

    friedman_result:
        Estatística, p-valor e decisão do teste de Friedman.

    posthoc_results:
        Comparações pareadas de Wilcoxon com correção de Holm.

    report_text:
        Relatório textual resumindo a análise.
    """

    scores_by_fold: pd.DataFrame
    ranks_by_fold: pd.DataFrame
    average_ranks: pd.DataFrame
    friedman_result: pd.DataFrame
    posthoc_results: pd.DataFrame
    report_text: str


def run_statistical_tests(
    fold_results: pd.DataFrame,
    *,
    metric: str = "f1_macro",
    alpha: float = 0.05,
    expected_folds: int = 10,
) -> StatisticalTestResults:
    """
    Executa a análise estatística dos resultados da validação cruzada.

    O procedimento consiste em:

    1. organizar os valores da métrica por fold e modelo;
    2. calcular os ranks por fold;
    3. aplicar o teste de Friedman;
    4. caso Friedman rejeite H0, aplicar Wilcoxon pareado;
    5. corrigir os p-valores das comparações pareadas pelo
       procedimento de Holm.

    Parameters
    ----------
    fold_results:
        DataFrame produzido pela validação cruzada.

    metric:
        Métrica principal utilizada na comparação.

    alpha:
        Nível de significância adotado.

    expected_folds:
        Quantidade esperada de folds.

    Returns
    -------
    StatisticalTestResults
        Tabelas e relatório da análise estatística.
    """

    _validate_fold_results(
        fold_results=fold_results,
        metric=metric,
        alpha=alpha,
        expected_folds=expected_folds,
    )

    scores_by_fold = _build_scores_table(
        fold_results=fold_results,
        metric=metric,
    )

    # Rank 1 significa melhor desempenho no fold.
    # Empates recebem o rank médio das posições empatadas.
    ranks_by_fold = scores_by_fold.rank(
        axis=1,
        ascending=False,
        method="average",
    )

    average_ranks = _calculate_average_ranks(
        scores_by_fold=scores_by_fold,
        ranks_by_fold=ranks_by_fold,
        metric=metric,
    )

    friedman_result = _run_friedman_test(
        scores_by_fold=scores_by_fold,
        metric=metric,
        alpha=alpha,
    )

    friedman_is_significant = bool(
        friedman_result.loc[0, "reject_h0"]
    )

    if friedman_is_significant:
        posthoc_results = _run_pairwise_wilcoxon_holm(
            scores_by_fold=scores_by_fold,
            alpha=alpha,
        )
    else:
        posthoc_results = _empty_posthoc_table()

    report_text = _build_statistical_report(
        metric=metric,
        alpha=alpha,
        scores_by_fold=scores_by_fold,
        average_ranks=average_ranks,
        friedman_result=friedman_result,
        posthoc_results=posthoc_results,
    )

    LOGGER.info(
        "Teste de Friedman concluído: estatística=%.6f, p=%.10g.",
        friedman_result.loc[0, "statistic"],
        friedman_result.loc[0, "p_value"],
    )

    LOGGER.info(
        "Decisão: %s.",
        friedman_result.loc[0, "decision"],
    )

    return StatisticalTestResults(
        scores_by_fold=scores_by_fold,
        ranks_by_fold=ranks_by_fold,
        average_ranks=average_ranks,
        friedman_result=friedman_result,
        posthoc_results=posthoc_results,
        report_text=report_text,
    )


def save_statistical_results(
    results: StatisticalTestResults,
    output_directory: str | Path,
) -> None:
    """
    Salva as tabelas e o relatório da análise estatística.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    results.scores_by_fold.to_csv(
        output_directory / "f1_scores_by_fold.csv",
        index=True,
        index_label="fold",
        float_format="%.10f",
        encoding="utf-8-sig",
    )

    results.ranks_by_fold.to_csv(
        output_directory / "f1_ranks_by_fold.csv",
        index=True,
        index_label="fold",
        float_format="%.4f",
        encoding="utf-8-sig",
    )

    results.average_ranks.to_csv(
        output_directory / "average_ranks.csv",
        index=False,
        float_format="%.10f",
        encoding="utf-8-sig",
    )

    results.friedman_result.to_csv(
        output_directory / "friedman_result.csv",
        index=False,
        float_format="%.10f",
        encoding="utf-8-sig",
    )

    results.posthoc_results.to_csv(
        output_directory / "posthoc_wilcoxon_holm.csv",
        index=False,
        float_format="%.10f",
        encoding="utf-8-sig",
    )

    report_path = output_directory / "statistical_report.txt"

    report_path.write_text(
        results.report_text,
        encoding="utf-8",
    )

    LOGGER.info(
        "Resultados estatísticos salvos em: %s",
        output_directory.resolve(),
    )


def _build_scores_table(
    *,
    fold_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Organiza a métrica em uma matriz fold x classificador.
    """

    model_order = (
        fold_results
        .groupby("model")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    scores_by_fold = (
        fold_results
        .pivot(
            index="fold",
            columns="model",
            values=metric,
        )
        .sort_index()
        .reindex(columns=model_order)
    )

    scores_by_fold.columns.name = None
    scores_by_fold.index.name = "fold"

    return scores_by_fold


def _calculate_average_ranks(
    *,
    scores_by_fold: pd.DataFrame,
    ranks_by_fold: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Calcula rank médio e medidas descritivas da métrica principal.
    """

    average_ranks = pd.DataFrame(
        {
            "model": scores_by_fold.columns,
            "average_rank": ranks_by_fold.mean(axis=0).to_numpy(),
            f"{metric}_mean": scores_by_fold.mean(axis=0).to_numpy(),
            f"{metric}_std": scores_by_fold.std(axis=0).to_numpy(),
            f"{metric}_median": scores_by_fold.median(axis=0).to_numpy(),
        }
    )

    return (
        average_ranks
        .sort_values(
            by=["average_rank", f"{metric}_mean"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def _run_friedman_test(
    *,
    scores_by_fold: pd.DataFrame,
    metric: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Aplica o teste não paramétrico de Friedman.
    """

    samples = [
        scores_by_fold[model].to_numpy()
        for model in scores_by_fold.columns
    ]

    statistic, p_value = friedmanchisquare(*samples)

    reject_h0 = bool(p_value < alpha)

    decision = (
        "Rejeitar H0"
        if reject_h0
        else "Não rejeitar H0"
    )

    interpretation = (
        "Há evidência de diferença estatisticamente significativa "
        "entre pelo menos dois classificadores."
        if reject_h0
        else
        "Não há evidência suficiente de diferença estatisticamente "
        "significativa entre os classificadores."
    )

    return pd.DataFrame(
        [
            {
                "test": "Friedman",
                "metric": metric,
                "number_of_models": scores_by_fold.shape[1],
                "number_of_folds": scores_by_fold.shape[0],
                "statistic": float(statistic),
                "p_value": float(p_value),
                "alpha": alpha,
                "reject_h0": reject_h0,
                "decision": decision,
                "interpretation": interpretation,
            }
        ]
    )


def _run_pairwise_wilcoxon_holm(
    *,
    scores_by_fold: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Executa comparações pareadas de Wilcoxon e aplica
    correção de Holm aos p-valores.
    """

    records: list[dict[str, int | float | str]] = []

    for model_a, model_b in combinations(
        scores_by_fold.columns,
        2,
    ):
        scores_a = scores_by_fold[model_a].to_numpy()
        scores_b = scores_by_fold[model_b].to_numpy()

        differences = scores_a - scores_b

        wins_a = int(np.sum(differences > 0))
        wins_b = int(np.sum(differences < 0))
        ties = int(np.sum(np.isclose(differences, 0.0)))

        # O Wilcoxon não pode ser aplicado quando todas as
        # diferenças são exatamente zero.
        if np.all(np.isclose(differences, 0.0)):
            statistic = 0.0
            p_value = 1.0
        else:
            wilcoxon_result = wilcoxon(
                scores_a,
                scores_b,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(wilcoxon_result.statistic)
            p_value = float(wilcoxon_result.pvalue)

        mean_difference = float(np.mean(differences))
        median_difference = float(np.median(differences))

        if mean_difference > 0:
            better_model_by_mean = model_a
        elif mean_difference < 0:
            better_model_by_mean = model_b
        else:
            better_model_by_mean = "Empate"

        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
                "mean_difference_a_minus_b": mean_difference,
                "median_difference_a_minus_b": median_difference,
                "wins_model_a": wins_a,
                "wins_model_b": wins_b,
                "ties": ties,
                "better_model_by_mean": better_model_by_mean,
            }
        )

    posthoc_results = pd.DataFrame(records)

    reject_h0, adjusted_p_values, _, _ = multipletests(
        posthoc_results["p_value_raw"].to_numpy(),
        alpha=alpha,
        method="holm",
    )

    posthoc_results["p_value_holm"] = adjusted_p_values
    posthoc_results["reject_h0_holm"] = reject_h0

    posthoc_results["decision_holm"] = np.where(
        posthoc_results["reject_h0_holm"],
        "Diferença significativa",
        "Diferença não significativa",
    )

    return (
        posthoc_results
        .sort_values(
            by=["p_value_holm", "model_a", "model_b"],
        )
        .reset_index(drop=True)
    )


def _empty_posthoc_table() -> pd.DataFrame:
    """
    Retorna a estrutura vazia da tabela pós-hoc.
    """

    return pd.DataFrame(
        columns=[
            "model_a",
            "model_b",
            "wilcoxon_statistic",
            "p_value_raw",
            "mean_difference_a_minus_b",
            "median_difference_a_minus_b",
            "wins_model_a",
            "wins_model_b",
            "ties",
            "better_model_by_mean",
            "p_value_holm",
            "reject_h0_holm",
            "decision_holm",
        ]
    )


def _build_statistical_report(
    *,
    metric: str,
    alpha: float,
    scores_by_fold: pd.DataFrame,
    average_ranks: pd.DataFrame,
    friedman_result: pd.DataFrame,
    posthoc_results: pd.DataFrame,
) -> str:
    """
    Gera um relatório textual da análise estatística.
    """

    statistic = friedman_result.loc[0, "statistic"]
    p_value = friedman_result.loc[0, "p_value"]
    decision = friedman_result.loc[0, "decision"]
    interpretation = friedman_result.loc[0, "interpretation"]

    lines = [
        "RELATÓRIO DE ANÁLISE ESTATÍSTICA",
        "=" * 40,
        "",
        f"Métrica principal: {metric}",
        f"Nível de significância: {alpha}",
        f"Quantidade de folds: {scores_by_fold.shape[0]}",
        f"Quantidade de classificadores: {scores_by_fold.shape[1]}",
        "",
        "TESTE DE FRIEDMAN",
        "-" * 40,
        f"Estatística: {statistic:.10f}",
        f"p-valor: {p_value:.10g}",
        f"Decisão: {decision}",
        f"Interpretação: {interpretation}",
        "",
        "RANKS MÉDIOS",
        "-" * 40,
    ]

    for row in average_ranks.itertuples(index=False):
        lines.append(
            f"{row.model}: rank médio={row.average_rank:.4f}, "
            f"F1 médio={row.f1_macro_mean:.4f}"
        )

    if not posthoc_results.empty:
        lines.extend(
            [
                "",
                "PÓS-HOC: WILCOXON COM CORREÇÃO DE HOLM",
                "-" * 40,
            ]
        )

        for row in posthoc_results.itertuples(index=False):
            lines.append(
                f"{row.model_a} x {row.model_b}: "
                f"p bruto={row.p_value_raw:.10g}, "
                f"p Holm={row.p_value_holm:.10g}, "
                f"{row.decision_holm}"
            )
    else:
        lines.extend(
            [
                "",
                "O pós-hoc não foi executado porque o teste de "
                "Friedman não rejeitou H0.",
            ]
        )

    lines.extend(
        [
            "",
            "RESSALVA METODOLÓGICA",
            "-" * 40,
            "Os resultados estatísticos devem ser interpretados no "
            "contexto da FMA Small e do protocolo experimental adotado. "
            "Os folds de uma validação cruzada não equivalem a conjuntos "
            "de dados totalmente independentes.",
        ]
    )

    return "\n".join(lines)


def _validate_fold_results(
    *,
    fold_results: pd.DataFrame,
    metric: str,
    alpha: float,
    expected_folds: int,
) -> None:
    """
    Valida a estrutura da tabela antes dos testes estatísticos.
    """

    required_columns = {
        "model",
        "fold",
        metric,
    }

    missing_columns = required_columns.difference(
        fold_results.columns
    )

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes em fold_results: "
            f"{sorted(missing_columns)}"
        )

    if fold_results.empty:
        raise ValueError("fold_results está vazio.")

    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar entre 0 e 1.")

    if expected_folds < 2:
        raise ValueError(
            "expected_folds deve ser igual ou superior a 2."
        )

    if fold_results[metric].isna().any():
        raise ValueError(
            f"Existem valores ausentes na métrica {metric}."
        )

    if not pd.api.types.is_numeric_dtype(
        fold_results[metric]
    ):
        raise TypeError(
            f"A métrica {metric} precisa ser numérica."
        )

    duplicated_results = fold_results.duplicated(
        subset=["model", "fold"]
    ).sum()

    if duplicated_results:
        raise ValueError(
            f"Foram encontrados {duplicated_results} resultados "
            "duplicados por modelo e fold."
        )

    number_of_models = fold_results["model"].nunique()

    if number_of_models < 3:
        raise ValueError(
            "O teste de Friedman exige pelo menos três grupos."
        )

    fold_counts = fold_results.groupby("model")["fold"].nunique()

    if not (fold_counts == expected_folds).all():
        raise ValueError(
            "Nem todos os modelos possuem a quantidade esperada "
            f"de folds:\n{fold_counts.to_string()}"
        )

    expected_fold_set = set(range(1, expected_folds + 1))

    for model_name, model_results in fold_results.groupby("model"):
        model_folds = set(model_results["fold"])

        if model_folds != expected_fold_set:
            raise ValueError(
                f"O modelo {model_name} não possui exatamente "
                f"os folds {sorted(expected_fold_set)}."
            )
