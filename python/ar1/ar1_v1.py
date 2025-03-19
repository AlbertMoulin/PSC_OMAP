import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def estimate_ar1_from_csv(csv_file):
    # Load CSV file with proper handling of quotes and delimiters
    data = pd.read_csv(csv_file, sep=',', header=None)

    # Convert data into numeric values
    data = data.apply(pd.to_numeric, errors='coerce')

    # Flatten the DataFrame into a single time series
    series = data.values.flatten()

    # Drop NaN values if any
    series = series[~np.isnan(series)]

    # Fit an AR(1) model
    model = ARIMA(series, order=(1, 0, 0))
    result = model.fit()

    # Extract and return AR(1) coefficient
    ar1_param = result.arparams[0]
    return ar1_param, series

def simulate_ar1(phi, n, initial_value, noise_std=0.1):
    simulated_series = np.zeros(n)
    simulated_series[0] = initial_value

    for t in range(1, n):
        simulated_series[t] = phi * simulated_series[t - 1] + np.random.normal(0, noise_std)

    return simulated_series

if __name__ == "__main__":
    csv_filename = "C:/Users/alex/Desktop/psc/ar1/data.csv"  # Change this to your actual CSV file path
    ar1_coefficient, original_series = estimate_ar1_from_csv(csv_filename)
    print(f"Estimated AR(1) coefficient: {ar1_coefficient:.4f}")

    # Simulate the AR(1) process using the estimated coefficient
    simulated_series = simulate_ar1(ar1_coefficient, len(original_series), original_series[0])

    # Plot the original and simulated series
    plt.figure(figsize=(10, 5))
    plt.plot(original_series, label="Original Series", linestyle='dashed')
    plt.plot(simulated_series, label="Simulated AR(1) Series")
    plt.legend()
    plt.title("Original vs Simulated AR(1) Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
