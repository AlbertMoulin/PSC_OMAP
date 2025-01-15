import csv

# Lire le fichier d'origine
with open(r'D:\Albert\Polytechnique\PSC\Code\data_dette\rates.csv', 'r', newline='', encoding='utf-8') as fichier_entree:
    lecteur = csv.reader(fichier_entree, delimiter=';')  # Le fichier d'origine utilise ";" comme séparateur
    lignes = []
    for ligne in lecteur:
        # Remplacer les virgules des décimales par des points dans chaque valeur
        nouvelle_ligne = [valeur.replace(',', '.') for valeur in ligne]
        lignes.append(nouvelle_ligne)

# Écrire dans un nouveau fichier avec le séparateur désiré
with open('ratesnew.csv', 'w', newline='', encoding='utf-8') as fichier_sortie:
    # Définir un écrivain CSV avec une virgule comme délimiteur
    ecrivain = csv.writer(fichier_sortie, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Écrire les lignes modifiées dans le nouveau fichier
    ecrivain.writerows(lignes)

print("Fichier CSV modifié avec succès.")
