import pandas as pd
import json
import glob
from pandas import json_normalize


# 📂 Chemin du dossier contenant les fichiers JSON
chemin_dossier = "C:/code/PSC_OMAP/Données_scrapping/json"  # Windows



# 🔎 Récupérer la liste de tous les fichiers JSON dans le dossier
fichiers_json = glob.glob(f"{chemin_dossier}/*.json")

# 📥 Liste pour stocker les données JSON
data_list = []

# 🔄 Boucle pour charger chaque fichier JSON
for fichier in fichiers_json:
    with open(fichier, "r", encoding="utf-8") as f:
        data = json.load(f)  # Charger le JSON en dictionnaire Python
        data_normalized = json_normalize(data)  # Aplatir le JSON
        data_list.append(data_normalized)  # Ajouter à la liste

# 📊 Convertir la liste en DataFrame pandas
df = pd.concat(data_list, ignore_index=True)

# 🔍 Afficher les premières lignes
print(df.head())

chemin_sauvegarde = "C:/code/PSC_OMAP/Données_scrapping"  # Windows

# 📥 Sauvegarde en CSV
df.to_csv(f"{chemin_sauvegarde}/donnees_fusionnees.csv", index=False, encoding="utf-8")

print(f"Fichier CSV enregistré à : {chemin_sauvegarde}.donnees_fusionnees.csv")


