import logging
from pathlib import Path

import pandas as pd

from src.visualizations import generate_visualizations


PROJECT_ROOT = Path(__file__).resolve().parent

FOLD_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "fold_results.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "predictions.csv"
)

AVERAGE_RANKS_PATH = (
    PROJECT_ROOT
    / "results"
    / "statistics"
    / "average_ranks.csv"
)

FIGURES_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

VISUALIZATION_TABLES_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "visualization"
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    required_files = [
        FOLD_RESULTS_PATH,
        PREDICTIONS_PATH,
        AVERAGE_RANKS_PATH,
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {file_path}"
            )

    fold_results = pd.read_csv(FOLD_RESULTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    average_ranks = pd.read_csv(AVERAGE_RANKS_PATH)

    class_metrics = generate_visualizations(
        fold_results=fold_results,
        predictions=predictions,
        average_ranks=average_ranks,
        figures_directory=FIGURES_DIRECTORY,
        tables_directory=VISUALIZATION_TABLES_DIRECTORY,
    )

    print(
        "\nVisualizações geradas em:",
        FIGURES_DIRECTORY.resolve(),
    )

    print(
        "Registros de métricas por classe:",
        len(class_metrics),
    )


if __name__ == "__main__":
    main()