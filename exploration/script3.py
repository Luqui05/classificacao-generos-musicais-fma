import pandas as pd

features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

print(features.shape)
print(features.columns[:10])
print(features.head())