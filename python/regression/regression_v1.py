import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Lire les données depuis le fichier texte
data = np.loadtxt('C:/Users/Alex/Desktop/psc/regression/data_x8.txt')

# Afficher la forme des données
print("Forme des données :", data.shape)

# Afficher les premières lignes des données
print("Données :")
print(data)

# Si vous voulez séparer les colonnes (par exemple, X et y pour une régression)
X = data[:, 8]  # Première colonne comme variable indépendante
y = data[:, 9]  # Deuxième colonne comme variable dépendante

# Exemple d'utilisation pour une régression linéaire
from sklearn.linear_model import LinearRegression

# Redimensionner X pour qu'il soit compatible avec scikit-learn
X = X.reshape(-1, 1)

# Appliquer la régression linéaire
model = LinearRegression()
model.fit(X, y)

# Faire des prédictions
y_pred = model.predict(X)

# Afficher les résultats
print("Coefficient de détermination R^2 :", model.score(X, y))

# Visualiser les résultats
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, y_pred, color='red', label='Régression linéaire')
plt.xlabel('Temps')
plt.ylabel('Valeurs')
plt.title('Régression linéaire sur un vecteur temporel')
plt.legend()
plt.show()

# Évaluer le modèle
r_squared = model.score(X_numeric, y)
print(f'Coefficient de détermination R^2: {r_squared}')

