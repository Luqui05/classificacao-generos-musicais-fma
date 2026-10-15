from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_DIRECTORY = PROJECT_ROOT / "data" / "fma_metadata"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fma_small_features.csv"


def main() -> None:
    tracks = pd.read_csv(
        METADATA_DIRECTORY / "tracks.csv",
        index_col=0,
        header=[0, 1],
    )
    features = pd.read_csv(
        METADATA_DIRECTORY / "features.csv",
        index_col=0,
        header=[0, 1, 2],
    )

    small = tracks[("set", "subset")] == "small"
    genres = tracks.loc[small, ("track", "genre_top")]

    valid_genres = genres.notna()
    genres = genres.loc[valid_genres]
    dataset = features.loc[genres.index].copy()

    dataset.columns = ["_".join(column).strip() for column in dataset.columns]
    dataset["genre"] = genres
    dataset.index.name = "track_id"

    if dataset.shape != (8000, 519):
        raise ValueError(f"Estrutura inesperada antes da exportação: {dataset.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=True)

    print(f"Base salva em: {OUTPUT_PATH}")
    print(dataset["genre"].value_counts())


if __name__ == "__main__":
    main()
