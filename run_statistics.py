import logging
from pathlib import Path

import pandas as pd

from src.statistical_tests import (
    run_statistical_tests,
    save_statistical_results,
)


PROJECT_ROOT = Path(__file__).resolve().parent

FOLD_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "fold_results.csv"
)

STATISTICS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "statistics"
)

METRIC = "f1_macro"
ALPHA = 0.05
EXPECTED_FOLDS = 10


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if not FOLD_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {FOLD_RESULTS_PATH}"
        )

    fold_results = pd.read_csv(FOLD_RESULTS_PATH)

    results = run_statistical_tests(
        fold_results=fold_results,
        metric=METRIC,
        alpha=ALPHA,
        expected_folds=EXPECTED_FOLDS,
    )

    save_statistical_results(
        results,
        STATISTICS_DIRECTORY,
    )

    print("\nTeste de Friedman:")
    print(results.friedman_result.to_string(index=False))

    print("\nRanks médios:")
    print(results.average_ranks.to_string(index=False))

    if not results.posthoc_results.empty:
        print("\nWilcoxon pareado com correção de Holm:")
        print(results.posthoc_results.to_string(index=False))


if __name__ == "__main__":
    main()
