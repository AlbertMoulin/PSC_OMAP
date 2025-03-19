import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# --- Chargement et préparation des données ---
csv_filename = "C:/Users/alex/Desktop/psc/ar1/data.csv"
data = pd.read_csv(csv_filename, header=None)

# Extraire la première colonne, convertir en numérique et supprimer les NaN
series = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()

# Conserver l'index d'origine sous forme continue
series.index = np.arange(len(series))

# --- Test de stationnarité ---
adf_stat, p_value, _, _, _, _ = adfuller(series)
print(f"ADF Statistic = {adf_stat:.3f}, p-value = {p_value:.3f}")

# Si la série n'est pas stationnaire, on utilisera ARIMA avec d=1 pour obtenir des prévisions sur l'échelle d'origine
if p_value > 0.05:
    print("La série n'est pas stationnaire. On utilisera ARIMA(6,1,0) pour obtenir des prévisions en valeurs réelles.")
    d = 1
else:
    print("La série est stationnaire. On utilisera ARIMA(6,0,0).")
    d = 0

# --- Découpage en 50% in-sample et 50% out-of-sample ---
split_index = len(series) // 2
training = series.iloc[:split_index].copy()   # in-sample
out_sample = series.iloc[split_index:].copy()   # out-of-sample

# --- Paramètres de prévision ---
prediction_horizon = 365  # nombre de points par bloc (1 an)
num_full_years = len(out_sample) // prediction_horizon  # nombre de blocs complets

# Préparer le tracé
plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Série d'origine", color="blue", linestyle="dashed")
plt.axvline(x=split_index, color="green", linestyle="--", label="Début out-of-sample")

# --- Boucle d'actualisation annuelle sur blocs complets ---
current_train = training.copy()  # jeu d'estimation courant

# Pour stocker les prévisions de chaque bloc
all_forecast = []
all_lower = []
all_upper = []
forecast_indices = []

for i in range(num_full_years):
    steps = prediction_horizon  # on travaille uniquement sur des blocs complets
    # Ré-estimation du modèle ARIMA (avec d fixé en fonction du test) sur current_train
    model = ARIMA(current_train, order=(6, d, 1),
                  enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit()

    # Prévision des 'steps' prochaines observations (les prévisions seront sur l'échelle d'origine)
    forecast_result = result.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    lower_bound = conf_int.iloc[:, 0]
    upper_bound = conf_int.iloc[:, 1]

    # Génération de l'axe x pour la prévision : suite de l'index de current_train
    last_idx = current_train.index[-1]
    f_idx = np.arange(last_idx + 1, last_idx + steps + 1)

    # Stockage des résultats pour ce bloc
    forecast_indices.append(f_idx)
    all_forecast.append(forecast_mean)
    all_lower.append(lower_bound)
    all_upper.append(upper_bound)

    # Actualisation : ajouter les observations réelles de ce bloc out-of-sample à current_train
    start = i * prediction_horizon
    end = start + steps
    actual_block = out_sample.iloc[start:end]
    current_train = pd.concat([current_train, actual_block], ignore_index=True)

# Rassembler les prévisions de tous les blocs
forecast_series = pd.concat(all_forecast, ignore_index=True)
lower_series = pd.concat(all_lower, ignore_index=True)
upper_series = pd.concat(all_upper, ignore_index=True)
forecast_index = np.concatenate(forecast_indices)

# --- Tracé des prévisions ---
plt.plot(forecast_index, forecast_series, label="Prévisions AR(6)", color="red", linewidth=2)
plt.fill_between(forecast_index, lower_series, upper_series, color="red", alpha=0.3, label="IC 95%")

plt.legend()
plt.title("Prévisions AR(6) avec actualisation annuelle (en valeurs réelles)")
plt.xlabel("Index temporel")
plt.ylabel("Valeur")
plt.show()
