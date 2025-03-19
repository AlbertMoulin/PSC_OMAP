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

def estimate_ar2_from_csv(csv_file):
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

    # Ajuster un modèle AR(2)
    model = ARIMA(series_diff, order=(2, 0, 0))
    result = model.fit()

    # Extraire et retourner les coefficients AR(1) et AR(2)
    ar2_params = result.arparams
    return ar2_params, series, series_diff

def simulate_ar2(phi1, phi2, n, initial_values, noise_std=0.1, num_simulations=100):
    """
    Simule plusieurs séries AR(2) donnée un coefficient phi1, phi2 et des valeurs initiales.
    """
    simulated_series = np.zeros((num_simulations, n))
    simulated_series[:, 0], simulated_series[:, 1] = initial_values[0], initial_values[1]

    for t in range(2, n):
        simulated_series[:, t] = phi1 * simulated_series[:, t - 1] + phi2 * simulated_series[:, t - 2] + np.random.normal(0, noise_std, num_simulations)

    return simulated_series

if __name__ == "__main__":
    csv_filename = "C:/Users/alex/Desktop/psc/ar1/data.csv"  # Change this to your actual CSV file path
    ar2_params, original_series, series_diff = estimate_ar2_from_csv(csv_filename)
    print(f"Estimated AR(2) coefficients: AR1 = {ar2_params[0]:.4f}, AR2 = {ar2_params[1]:.4f}")

    # Simuler plusieurs séries AR(2) en utilisant les coefficients estimés
    num_simulations = 2000  # Nombre de simulations à réaliser
    initial_values = [series_diff[0], series_diff[1]]  # Utilisation des deux premières valeurs pour initialiser la simulation
    simulated_diff = simulate_ar2(ar2_params[0], ar2_params[1], len(series_diff), initial_values, num_simulations=num_simulations)

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
    plt.plot(mean_simulated_series, label="Simulated AR(2) Mean", color='red', linestyle='-', linewidth=2)
    plt.fill_between(range(len(original_series)), lower_bound, upper_bound, color='red', alpha=0.2, label="90% Confidence Interval")

    plt.legend()
    plt.title("Original Series vs Simulated AR(2) with 90% Confidence Interval")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
