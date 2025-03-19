import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import os

# Lire les données depuis le fichier texte
data = np.loadtxt('C:/Users/Alex/Desktop/psc/regression/data_x8.txt')

# Vérifier la forme des données
print("Shape de data :", data.shape)  # Doit afficher (141, 10) si correct

# Créer la variable temps sous forme de nombres (jours écoulés)
start_date = datetime.date(2007,1,11)
dates = np.array([start_date + datetime.timedelta(weeks=i) for i in range(data.shape[0])])
dates_numeric = np.array([(d - start_date).days for d in dates]).reshape(-1, 1)  # Conversion en jours écoulés

# Vérifier la taille de dates_numeric
print("Shape de dates_numeric :", dates_numeric.shape)

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

    # Vérification des dimensions avant le fit
    print(f"Variable {i+1} - Shape : {variable.shape}, Dates Shape: {dates_numeric.shape}")

    # Ajuster le modèle de régression linéaire
    model.fit(dates_numeric, variable)

    # Faire des prédictions
    predictions = model.predict(dates_numeric)

    # Afficher les résultats
    print(f"Variable {i + 1}:")
    print(f"  Coefficient (pente) : {model.coef_[0]}")
    print(f"  Intercept : {model.intercept_}")
    print(f"  R^2 : {model.score(dates_numeric, variable)}")
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
