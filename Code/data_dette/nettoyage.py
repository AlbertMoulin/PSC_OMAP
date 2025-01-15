import pandas as pd

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(r'Code\data_dette\ratesnew.csv')

# # Afficher un aperçu des données
# print(df.head())

# Afficher le nombre de valeurs manquantes par colonne
print(df.isnull().sum())

# # Identifier les lignes contenant des valeurs manquantes
# print(df[df.isnull().any(axis=1)])

