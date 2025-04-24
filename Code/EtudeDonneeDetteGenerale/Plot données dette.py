import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les données depuis le fichier CSV
data = pd.read_excel('C:\code\PSC_OMAP\Code\EtudeDonneeDetteGenerale\BDF_Tresor_2004_2008_maturites10_quotidien.xlsx')

# Afficher les premières lignes du dataframe pour vérifier le chargement des données
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
# Convertir toutes les autres colonnes en valeurs numériques
for col in data.columns[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dummy variables to choose which plots to display
plot_all_maturities = True
plot_selected_maturities = False
plot_selected_maturities_postcovid = False

if plot_all_maturities:
    # Créer des subplots
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    axes = axes.flatten()

    # Tracer les données pour chaque maturité
    for i, mat in enumerate(data.columns[1:]):
        print(mat)
        axes[i].plot(data['Date'], data[mat], linestyle='-')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Valeur')
        axes[i].set_title(f'Maturité: {mat}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

if plot_selected_maturities:
    # Tracer les maturités 1, 5, 10 et 30 sur le même graphique avec des lignes plus fines
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data[1], label='Maturité 1', linewidth=0.5)
    plt.plot(data['Date'], data[5], label='Maturité 5', linewidth=0.5)
    plt.plot(data['Date'], data[10], label='Maturité 10', linewidth=0.5)
    plt.plot(data['Date'], data[30], label='Maturité 30', linewidth=0.5)

    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.title('Maturités 1, 5, 10 et 30')
    plt.legend()
    plt.grid(True)
    plt.show()


if plot_selected_maturities_postcovid:
    # Tracer les maturités 1, 5, 10 et 30 sur le même graphique avec des lignes plus fines
    data_postcovid = data[data['Date'] >= '2020-01-01']
    plt.figure(figsize=(10, 6))
    plt.plot(data_postcovid['Date'], data_postcovid[1], label='Maturité 1', linewidth=0.5)
    plt.plot(data_postcovid['Date'], data_postcovid[5], label='Maturité 5', linewidth=0.5)
    plt.plot(data_postcovid['Date'], data_postcovid[10], label='Maturité 10', linewidth=0.5)
    plt.plot(data_postcovid['Date'], data_postcovid[30], label='Maturité 30', linewidth=0.5)

    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.title('Maturités 1, 5, 10 et 30')
    plt.legend()
    plt.grid(True)
    plt.show()