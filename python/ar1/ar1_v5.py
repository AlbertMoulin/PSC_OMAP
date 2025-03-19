import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Fonction pour tester si la série est stationnaire avec ADF
def test_stationarity(series):
    adf_stat, p_value, _, _, _, _ = adfuller(series)
    return adf_stat, p_value

# Charger les données
data = pd.read_csv('C:/Users/alex/Desktop/psc/ar1/data_x5.csv', header=None)
#data = np.loadtxt('C:/Users/Alex/Desktop/psc/ar1/data_x5.txt')
series = data.iloc[:, 0]  # Utiliser uniquement la 1re colonne

# Tester la stationnarité
adf_stat, p_value = test_stationarity(series)
print(f"ADF Statistic = {adf_stat:.3f}, p-value = {p_value:.3f}")

# Si p-value est > 0.05, la série est non stationnaire et nécessiterait une différenciation
d = 1 if p_value > 0.05 else 0

# Entrainement et prédiction AR(1)
train_size = int(len(series) * 0.5)  # Diviser en 2 parties (50% de données pour entraînement)
train, test = series[:train_size], series[train_size:]

# Model AR(1) sur la partie entraînement
model = ARIMA(train, order=(1, d, 0))  # AR(1)
result = model.fit()

# Afficher le résumé du modèle
print(result.summary())

# Générer les prédictions sur la partie test
predictions = result.forecast(steps=len(test))

# Tracer la série réelle et les prédictions AR(1)
plt.figure(figsize=(12, 6))
plt.plot(series.index[:train_size], train, label='Train', color='blue')
plt.plot(series.index[train_size:], test, label='Test', color='orange')
plt.plot(series.index[train_size:], predictions, label='AR(1) Predictions', color='green', linestyle='--')
plt.legend()
plt.title("AR(1) Model Predictions vs Real Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
