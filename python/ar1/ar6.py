import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Charger les données depuis le CSV
csv_filename = "C:/Users/alex/Desktop/psc/ar1/data.csv"
data = pd.read_csv(csv_filename, header=None)

# Extraire la première colonne, convertir en numérique et supprimer les NaN
series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna().reset_index(drop=True)

# Définir la partie in-sample (première moitié) et out-of-sample (seconde moitié)
split_index = len(series) // 2
training = series.iloc[:split_index].copy()  # in-sample initial
out_sample = series.iloc[split_index:].copy()  # out-of-sample

# Paramètres de prévision
prediction_horizon = 365   # horizon de prévision (chaque "année")
num_years = len(out_sample) // prediction_horizon  # nombre d'années à prévoir

# Pour stocker les prévisions et intervalles
all_forecast = []
all_lower = []
all_upper = []
forecast_indices = []

# Initialisation de la série d'entraînement courante
current_train = training.copy()

for i in range(num_years):
    # Ré-estimer le modèle AR(6) sur les données disponibles
    model = ARIMA(current_train, order=(6, 0, 0))
    result = model.fit()  # Utilisation de la méthode par défaut 'mle'

    # Prévoir les prochains prediction_horizon points
    forecast_result = result.get_forecast(steps=prediction_horizon)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    lower_bound = conf_int.iloc[:, 0]
    upper_bound = conf_int.iloc[:, 1]

    # Stocker les indices pour le tracé
    indices = np.arange(len(current_train), len(current_train) + prediction_horizon)
    forecast_indices.append(indices)

    # Stocker les prévisions et intervalles
    all_forecast.append(forecast_mean)
    all_lower.append(lower_bound)
    all_upper.append(upper_bound)

    # Récupérer les observations réelles out-of-sample pour cette période
    start = i * prediction_horizon
    end = start + prediction_horizon
    actual_year = out_sample.iloc[start:end]

    # Actualiser la série d'entraînement en ajoutant les observations réelles
    current_train = pd.concat([current_train, actual_year], ignore_index=True)

# Combiner les résultats de prévision
forecast_series = pd.concat(all_forecast, ignore_index=True)
lower_series = pd.concat(all_lower, ignore_index=True)
upper_series = pd.concat(all_upper, ignore_index=True)
forecast_index = np.concatenate(forecast_indices)

# Tracé des résultats
plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Série d'origine", color="blue", linestyle="dashed")
plt.axvline(x=split_index, color="green", linestyle="--", label="Début out-of-sample")
plt.plot(forecast_index, forecast_series, label="Prévision AR(6)", color="red", linewidth=2)
plt.fill_between(forecast_index, lower_series, upper_series, color="red", alpha=0.3, label="Intervalle de confiance 95%")
plt.legend()
plt.title("Prévision AR(6) avec actualisation annuelle")
plt.xlabel("Index temporel")
plt.ylabel("Valeur")
plt.show()
