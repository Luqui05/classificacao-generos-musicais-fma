import pandas as pd

tracks = pd.read_csv("tracks.csv", index_col=0, header=[0, 1])

print(tracks.shape)
print(tracks.columns)
print(tracks[("set", "subset")].value_counts())
print(tracks[("track", "genre_top")].value_counts())