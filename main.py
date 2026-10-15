import logging
from pathlib import Path

from src.cross_validation import (
    run_cross_validation,
    save_cross_validation_results,
)
from src.data_loader import load_dataset, save_dataset_reports
from src.models import build_models


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "fma_small_features.csv"
TABLES_DIRECTORY = PROJECT_ROOT / "results" / "tables"

N_SPLITS = 10
RANDOM_STATE = 42


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    dataset = load_dataset(
        DATASET_PATH,
        drop_duplicate_samples=True,
    )

    save_dataset_reports(
        dataset,
        TABLES_DIRECTORY,
    )

    models = build_models()

    cross_validation_results = run_cross_validation(
        X=dataset.X,
        y=dataset.y,
        track_ids=dataset.track_ids,
        models=models,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    save_cross_validation_results(
        cross_validation_results,
        TABLES_DIRECTORY,
    )

    print("\nResumo das métricas:")
    print(
        cross_validation_results.metrics_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
