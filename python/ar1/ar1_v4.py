import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(series):
    """
    Effectue le test Augmented Dickey-Fuller pour vérifier la stationnarité de la série temporelle.
    """
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("La série est stationnaire.")
    else:
        print("La série n'est pas stationnaire.")

def estimate_ar1_from_csv(csv_file):
    # Charger le fichier CSV et sélectionner uniquement la première colonne
    data = pd.read_csv(csv_file, sep=',', header=None)

    # Sélectionner uniquement la première colonne (index 0)
    series = data.iloc[:, 0].values

    # Convertir les données en valeurs numériques
    series = pd.to_numeric(series, errors='coerce')

    # Supprimer les valeurs NaN si présentes
    series = series[~np.isnan(series)]

    # Vérifier la stationnarité de la série
    check_stationarity(series)

    # Si la série n'est pas stationnaire, la différencier
    if adfuller(series)[1] >= 0.05:  # p-value >= 0.05, série non stationnaire
        print("La série n'est pas stationnaire, on va la différencier.")
        series_diff = np.diff(series)  # Différenciation de la série
    else:
        series_diff = series  # Série déjà stationnaire

    # Ajuster un modèle AR(1)
    model = ARIMA(series_diff, order=(1, 0, 0))
    result = model.fit()

    # Extraire et retourner le coefficient AR(1)
    ar1_param = result.arparams[0]
    return ar1_param, series, series_diff

def simulate_ar1(phi, n, initial_value, noise_std=0.1, num_simulations=100):
    """
    Simule plusieurs séries AR(1) donnée un coefficient phi et une valeur initiale.
    """
    simulated_series = np.zeros((num_simulations, n))
    simulated_series[:, 0] = initial_value

    for t in range(1, n):
        simulated_series[:, t] = phi * simulated_series[:, t - 1] + np.random.normal(0, noise_std, num_simulations)

    return simulated_series

if __name__ == "__main__":
    csv_filename = "C:/Users/alex/Desktop/psc/ar1/data_x5.csv"  # Change this to your actual CSV file path
    ar1_coefficient, original_series, series_diff = estimate_ar1_from_csv(csv_filename)
    print(f"Estimated AR(1) coefficient: {ar1_coefficient:.4f}")

    # Simuler plusieurs séries AR(1) en utilisant le coefficient estimé
    num_simulations = 5000  # Nombre de simulations à réaliser
    simulated_diff = simulate_ar1(ar1_coefficient, len(series_diff), series_diff[0], num_simulations=num_simulations)

    # Réintégrer la différence pour reconstituer la série originale simulée
    simulated_series = np.cumsum(simulated_diff, axis=1) + original_series[0]

    # Assurer que la série simulée a la même taille que la série originale
    if simulated_series.shape[1] != len(original_series):
        if simulated_series.shape[1] > len(original_series):
            simulated_series = simulated_series[:, :len(original_series)]
        else:
            # Compléter avec la dernière valeur si la simulation est trop petite
            last_value = simulated_series[:, -1].reshape(-1, 1)
            simulated_series = np.hstack([simulated_series, np.tile(last_value, (1, len(original_series) - simulated_series.shape[1]))])

    # Calculer la moyenne et les percentiles pour l'intervalle de confiance à 90%
    mean_simulated_series = np.mean(simulated_series, axis=0)
    lower_bound = np.percentile(simulated_series, 5, axis=0)  # 5% percentile (borne inférieure)
    upper_bound = np.percentile(simulated_series, 95, axis=0)  # 95% percentile (borne supérieure)

    # Tracer la série originale, la série simulée moyenne et l'intervalle de confiance
    plt.figure(figsize=(10, 5))
    plt.plot(original_series, label="Original Series", linestyle='dashed', color='blue')
    plt.plot(mean_simulated_series, label="Simulated AR(1) Mean", color='red', linestyle='-', linewidth=2)
    plt.fill_between(range(len(original_series)), lower_bound, upper_bound, color='red', alpha=0.2, label="90% Confidence Interval")

    plt.legend()
    plt.title("Original Series vs Simulated AR(1) with 90% Confidence Interval")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
