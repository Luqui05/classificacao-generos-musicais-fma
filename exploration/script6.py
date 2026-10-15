import pandas as pd

features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

columns_df = features.columns.to_frame(index=False)

columns_df.columns = ["feature_group", "statistic", "dimension"]

print(columns_df.head(30))

summary = (
    columns_df
    .groupby(["feature_group", "statistic"])
    .size()
    .reset_index(name="n_columns")
)

print(summary)

summary.to_csv("fma_features_summary.csv", index=False)