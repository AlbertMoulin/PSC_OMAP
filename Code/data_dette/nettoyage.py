import pandas as pd

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(r'Code\data_dette\rates.csv')


# pour chaque ligne vérifier si la longueur de la ligne est 8
# si ce n'est pas le cas, afficher la ligne
for index, row in df.iterrows():
    if len(row) != 8:
        print(row)


