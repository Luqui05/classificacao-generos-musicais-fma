import pandas as pd

features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

print("Formato:", features.shape)

print("\nGrupos de features:")
print(features.columns.get_level_values("feature").unique())

print("\nEstatísticas utilizadas:")
print(features.columns.get_level_values("statistics").unique())

print("\nQuantidade de colunas por grupo de feature:")
print(features.columns.get_level_values("feature").value_counts())

print("\nQuantidade de colunas por estatística:")
print(features.columns.get_level_values("statistics").value_counts())