import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Lire les données depuis le fichier texte
data = np.loadtxt('C:/Users/Alex/Desktop/psc/regression/data_x8.txt')

# Vérifier la forme des données
print("Forme des données :", data.shape)  # Doit afficher (nombre_de_lignes, 10)

# Créer la variable temps (indices des lignes)
temps = np.arange(data.shape[0]).reshape(-1, 1)  # Temps = [0, 1, 2, ..., n-1]

# liste coef
Lcoef=[]

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Boucle sur chaque variable (colonne)
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
    plt.figure(figsize=(8, 4))
    plt.scatter(temps, variable, color='blue', label='Données réelles')
    plt.plot(temps, predictions, color='red', label='Régression linéaire')
    plt.xlabel('Temps (indice)')
    plt.ylabel(f'Variable {i + 1}')
    plt.title(f'Régression linéaire pour la variable {i + 1}')
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 4))
plt.scatter([0,1,2,3,4,5,6,7,8,9], Lcoef, color='blue', label='')
plt.plot([0,1,2,3,4,5,6,7,8,9], Lcoef, color='red', label='')
plt.xlabel('indice')
plt.ylabel(f'valeur coef')
plt.title(f'coef')
plt.legend()
plt.show()