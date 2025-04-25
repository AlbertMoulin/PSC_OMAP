##Script pour comarer les facteurs estimés aux taux d'intérêts 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def load_data():
    ########### Facteurs:
       # Lire les données depuis le fichier texte
    array_factors = np.loadtxt('Code/Regressions_nickbloom/data_regression/mu_dd_CANFCPv2_FS10_31.txt')

    # Vérifier la forme des données
    print("Forme des données :", array_factors.shape)  # Doit afficher (nombre_de_lignes, 10)

    # Créer la variable temps sous forme de dates
    start_date = datetime.date(2007, 1, 3)
    dates = np.array([start_date + datetime.timedelta(weeks=i) for i in range(array_factors.shape[0])])


    data_factors=pd.DataFrame(array_factors)
    data_factors['Date']= dates
    data_factors['Date']=pd.to_datetime(data_factors['Date'])
    #renommer colonnes:
    data_factors.columns= [f"factor_{i}" for i in range(10)]+["Date"]




    ########### Taux d'intérêt:
    # Charger les données depuis le fichier CSV
    data_rates = pd.read_excel('Code/Regressions_nickbloom/data_regression/BDF_Tresor_2004_2008_maturites10_quotidien.xlsx')


    # Afficher les premières lignes du dataframe pour vérifier le chargement des données
    data_rates['Date'] = pd.to_datetime(data_rates['Date'], errors='coerce')
    # Convertir toutes les autres colonnes en valeurs numériques
    for col in data_rates.columns[1:]:
        data_rates[col] = pd.to_numeric(data_rates[col], errors='coerce')
    data_rates.columns= [f"maturity_{i}" for i in [1]]+["Date"]

    

    print(data_factors.head())
    print(data_rates.head())

    data_combined = pd.merge(data_factors, data_rates, on='Date')
    return data_combined

data_combined = load_data()
print(data_combined.head())