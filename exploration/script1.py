import pandas as pd

tracks = pd.read_csv("tracks.csv", index_col=0, header=[0, 1])
features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

# Seleciona apenas faixas da subset small
small = tracks[("set", "subset")] == "small"

# Rótulo: gênero principal
y = tracks.loc[small, ("track", "genre_top")]

# Features correspondentes às mesmas músicas
X = features.loc[y.index]

# Remove registros sem gênero, se houver
mask = y.notna()
X = X.loc[mask]
y = y.loc[mask]

print(X.shape)
print(y.shape)
print(y.value_counts())