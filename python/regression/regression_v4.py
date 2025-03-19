import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import os

# Lire les données depuis le fichier texte
data = np.loadtxt('C:/Users/Alex/Desktop/psc/regression/data_x8.txt')

# Vérifier la forme des données
print("Forme des données :", data.shape)  # Doit afficher (nombre_de_lignes, 10)

# Créer la variable temps sous forme de dates
start_date = datetime.date(2007,1,11)
dates = np.array([start_date + datetime.timedelta(weeks=i) for i in range(data.shape[0])])

# Filtrer les données à partir de 2022
start_2022_index = next(i for i, date in enumerate(dates) if date.year >= 2022)
dates = dates[start_2022_index:]
data = data[start_2022_index:]
temps = np.arange(len(dates)).reshape(-1, 1)

# Liste pour stocker les coefficients
Lcoef = []

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Créer un dossier pour enregistrer les figures
output_dir = "C:/Users/Alex/Desktop/psc/regression/results"
os.makedirs(output_dir, exist_ok=True)

# Boucle sur chaque variable (colonne)
plt.figure(figsize=(12, 8))
for i in range(data.shape[1]):
    # Extraire la variable courante
    variable = data[:, i]

    # Ajuster le modèle de régression linéaire
    model.fit(temps, variable)

    # Faire des prédictions
    predictions = model.predict(temps)

    # Afficher les résultats
    print(f"Variable {i + 1}:")
    print(f"  Coefficient (pente) : {model.coef_[0]}")
    print(f"  Intercept : {model.intercept_}")
    print(f"  R^2 : {model.score(temps, variable)}")
    Lcoef.append(model.coef_[0])

    # Tracer les résultats
    plt.subplot(2, 5, i+1)
    plt.scatter(dates, variable, color='blue', label='Données réelles')
    plt.plot(dates, predictions, color='red', label='Régression linéaire')
    plt.xlabel('Temps (dates)')
    plt.ylabel(f'Variable {i + 1}')
    plt.title(f'Variable {i + 1}')
    plt.xticks(rotation=45)
    plt.legend()

    # Sauvegarder chaque graphique
    plt.figure(figsize=(8, 4))
    plt.scatter(dates, variable, color='blue', label='Données réelles')
    plt.plot(dates, predictions, color='red', label='Régression linéaire')
    plt.xlabel('Temps (dates)')
    plt.ylabel(f'Variable {i + 1}')
    plt.title(f'Régression linéaire pour la variable {i + 1}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'variable_{i+1}.png'))
    plt.close()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_variables.png'))
plt.show()

# Tracer l'évolution des coefficients
plt.figure(figsize=(8, 4))
plt.scatter(range(len(Lcoef)), Lcoef, color='blue', label='Coefficients')
plt.plot(range(len(Lcoef)), Lcoef, color='red')
plt.xlabel('Indice')
plt.ylabel('Valeur du coefficient')
plt.title('Évolution des coefficients')
plt.legend()
plt.savefig(os.path.join(output_dir, 'coefficients.png'))
plt.show()
