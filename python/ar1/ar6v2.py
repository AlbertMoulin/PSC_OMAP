import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# --- Chargement et préparation des données ---
csv_filename = "C:/Users/alex/Desktop/psc/ar1/data_x5.csv"
data = pd.read_csv(csv_filename, header=None)

# Extraire la première colonne, convertir en numérique et supprimer les NaN
series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
# Conserver l'index d'origine sous forme continue
series.index = np.arange(len(series))

# --- Vérification de la stationnarité ---
adf_stat, p_value, _, _, _, _ = adfuller(series)
print(f"ADF Statistic = {adf_stat:.3f}, p-value = {p_value:.3f}")
if p_value > 0.05:
    print("La série n'est pas stationnaire, on applique une différenciation.")
    series = series.diff().dropna()
    # Réassigner un nouvel index continu
    series.index = np.arange(len(series))
else:
    print("La série est stationnaire.")

# --- Découpage en 50% in-sample et 50% out-of-sample ---
split_index = len(series) // 2
training = series.iloc[:split_index].copy()   # in-sample
out_sample = series.iloc[split_index:].copy()   # out-of-sample

# --- Paramètres de prévision ---
prediction_horizon = 365  # nombre de points par bloc (1 an)
num_full_years = len(out_sample) // prediction_horizon  # nombre de blocs complets

# Préparer le tracé
plt.figure(figsize=(12, 6))
# Tracé des données réelles : in-sample et out-of-sample affichés séparément pour plus de lisibilité
plt.plot(training.index, training, label="Données in-sample réelles", color="blue", linestyle="dashed")
plt.plot(out_sample.index, out_sample, label="Données out-of-sample réelles", color="blue", marker="o", linestyle="None")
plt.axvline(x=split_index, color="green", linestyle="--", label="Début out-of-sample")

# --- Boucle d'actualisation annuelle sur blocs complets ---
current_train = training.copy()  # initialisation de l'in-sample courant

# Pour stocker les prévisions de chaque bloc
all_forecast = []
all_lower = []
all_upper = []
forecast_indices = []

for i in range(num_full_years):
    steps = prediction_horizon  # on travaille sur des blocs complets uniquement
    # Ré-estimer le modèle AR(6) sur les données disponibles
    model = ARIMA(current_train, order=(6, 0, 0),
                  enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit()

    # Prévoir les 'steps' prochains points
    forecast_result = result.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    lower_bound = conf_int.iloc[:, 0]
    upper_bound = conf_int.iloc[:, 1]

    # Génération de l'axe x pour la prévision (continuation de l'index de current_train)
    last_idx = current_train.index[-1]
    f_idx = np.arange(last_idx + 1, last_idx + steps + 1)

    # Stockage des résultats pour ce bloc
    forecast_indices.append(f_idx)
    all_forecast.append(forecast_mean)
    all_lower.append(lower_bound)
    all_upper.append(upper_bound)

    # Actualisation : on ajoute les observations réelles du bloc out-of-sample courant
    start = i * prediction_horizon
    end = start + steps
    actual_block = out_sample.iloc[start:end]
    current_train = pd.concat([current_train, actual_block], ignore_index=True)

# Rassembler les prévisions de tous les blocs
forecast_series = pd.concat(all_forecast, ignore_index=True)
lower_series = pd.concat(all_lower, ignore_index=True)
upper_series = pd.concat(all_upper, ignore_index=True)
forecast_index = np.concatenate(forecast_indices)

# Tracé des prévisions et des intervalles de confiance pour chaque bloc annuel
plt.plot(forecast_index, forecast_series, label="Prévisions AR(6)", color="red", linewidth=2)
plt.fill_between(forecast_index, lower_series, upper_series, color="red", alpha=0.3, label="IC 95%")

plt.legend()
plt.title("Prévisions AR(6) avec actualisation annuelle (blocs de 1 an)")
plt.xlabel("Index temporel")
plt.ylabel("Valeur (sur l'échelle de la série ou de ses différences)")
plt.show()
