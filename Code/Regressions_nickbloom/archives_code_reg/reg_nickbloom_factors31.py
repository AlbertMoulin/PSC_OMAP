#part des facteurs estimé et d'une interpolation de l'indice de Nick Bloom. Fait la régression.
#Utilise des contrôles éventuels

import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import os

def load_data_reg():

    ###Récupérer données Nick Bloom:

    # Load the Excel file
    file_path = 'Code/Regressions_nickbloom/data_regression/Europe_Policy_Uncertainty_Data (1).xlsx'
    data_NB = pd.read_excel(file_path, engine='openpyxl')

    # Drop the last line of the data
    data_NB = data_NB[:-1]
    # Print the last value of the column "Year"
    print("Last year in the data:", data_NB['Year'].iloc[-1])

    # Create a new column that indicates the date of each line as the first day of each month
    data_NB['Date'] = pd.to_datetime(data_NB[['Year', 'Month']].assign(DAY=1))


    ###Récupérer Facteurs:


    # Lire les données depuis le fichier texte
    array_factors = np.loadtxt('Code/Regressions_nickbloom/data_regression/mu_dd_CANFCPv2_FS10_31.txt')

    # Vérifier la forme des données
    print("Forme des données :", array_factors.shape)  # Doit afficher (nombre_de_lignes, 10)

    # Créer la variable temps sous forme de dates
    start_date = datetime.date(2007, 1, 3)
    dates = np.array([start_date + datetime.timedelta(weeks=i) for i in range(array_factors.shape[0])])

    # Convertir les dates en indices pour la régression linéaire
    temps = np.arange(array_factors.shape[0]).reshape(-1, 1)  # [0, 1, 2, ..., n-1]

    data_factors=pd.DataFrame(array_factors)
    data_factors['Date']= dates
    data_factors['Date']=pd.to_datetime(data_factors['Date'])

    # # Liste pour stocker les coefficients
    # Lcoef = []

    # # Initialiser le modèle de régression linéaire
    # model = LinearRegression()

    # # Créer un dossier pour enregistrer les figures
    # output_dir = "C:/Users/Alex/Desktop/psc/regression/results"
    # os.makedirs(output_dir, exist_ok=True)


    ###Récupérer control vars
        

    ##Policy rate 


    file_path = 'Code/Regressions_nickbloom/data_regression/ECB Data Portal_20250319150056.csv'
    data_ECB = pd.read_csv(file_path)
    data_ECB.columns=['Date',"Time_period","Deposit_rate"]
    data_ECB['Date'] = pd.to_datetime(data_ECB['Date'])


    ## inflation France

    file_path = 'Code/Regressions_nickbloom/data_regression/ECB Data Portal_inflation_fr.csv'
    data_infl_fr = pd.read_csv(file_path)
    data_infl_fr.columns=['Date',"Time_period","Inflation_FR"]
    data_infl_fr['Date'] = pd.to_datetime(data_infl_fr['Date'])

    ## inflation expectations

    file_path = 'Code/Regressions_nickbloom/data_regression/ECB Data Portal_inflationexpect_lt.csv'
    data_infl_exp_lt = pd.read_csv(file_path)
    data_infl_exp_lt.columns=['Date',"Time_period","Inflation_forecast_LT"]
    data_infl_exp_lt['Date'] = pd.to_datetime(data_infl_exp_lt['Date'])

    ##euribor 3 months

    file_path = 'Code/Regressions_nickbloom/data_regression/ECB Data Portal_euribor3months.csv'
    data_euribor3 = pd.read_csv(file_path)
    data_euribor3.columns=['Date',"Time_period","EURIBOR"]
    data_euribor3['Date'] = pd.to_datetime(data_euribor3['Date'])

    ##Dette/PIB

    file_path = 'Code/Regressions_nickbloom/data_regression/ECB Data Portal_debt_gdp_france.csv'
    data_debt_gdp_FR = pd.read_csv(file_path)
    data_debt_gdp_FR.columns=['Date',"Time_period","Debt_GDP"]
    data_debt_gdp_FR['Date'] = pd.to_datetime(data_debt_gdp_FR['Date'])


    ### Process données en 1 dataframe

    ### Merge data_ECB and data_NB with data_factors based on the closest date

    #set dates to indexes
    data_NB.set_index('Date', inplace=True)
    data_factors.set_index('Date', inplace = True)
    data_ECB.set_index('Date', inplace=True)
    data_debt_gdp_FR.set_index('Date', inplace=True)
    data_euribor3.set_index('Date', inplace=True)
    data_infl_exp_lt.set_index('Date', inplace=True)
    data_infl_fr.set_index('Date', inplace=True)


    #select dates frm 2003:
    data_ECB=data_ECB[data_ECB.index>np.datetime64("2001-01-01")]

    # Interpolate data_ECB and data_NB to match the dates in data_factors
    data_NB = data_NB.reindex(data_ECB.index)
    data_factors = data_factors.reindex(data_ECB.index)
    data_ECB = data_ECB.reindex(data_ECB.index)
    data_debt_gdp_FR = data_debt_gdp_FR.reindex(data_ECB.index)
    data_euribor3 = data_euribor3.reindex(data_ECB.index)
    data_infl_exp_lt = data_infl_exp_lt.reindex(data_ECB.index)
    data_infl_fr = data_infl_fr.reindex(data_ECB.index)

    # # Interpoler les valeurs manquantes
    data_NB.interpolate(method='time', inplace=True)
    data_factors.interpolate(method='time', inplace=True)
    data_ECB.interpolate(method='time', inplace=True)
    data_debt_gdp_FR.interpolate(method='time', inplace=True)
    data_euribor3.interpolate(method='time', inplace=True)
    data_infl_exp_lt.interpolate(method='time', inplace=True)
    data_infl_fr.interpolate(method='time', inplace=True)



    # Concatenate the datasets
    #combined_data = pd.concat([data_factors, data_ECB['Deposit-rate'], data_NB, d], axis=1)

    Data_list = [data_NB, data_factors,data_ECB,data_infl_fr,data_infl_exp_lt,data_euribor3,data_debt_gdp_FR]

    combined_data=pd.concat([data for data in Data_list], axis =1)

    for data in Data_list:
        print(data.head())

    return combined_data



combined_data= load_data_reg()

# # choose only values for which we have data
combined_data=combined_data[combined_data.index> np.datetime64("2006-12-31")]


# #Interpolate all values on a daily basis
combined_data.interpolate(inplace=True)    

combined_data=combined_data[combined_data.index >np.datetime64("2007-01-02")]



# Print the combined data to verify
print(combined_data.head())

######################### 1st regression: monthly basis ############################################

#choose the dataset
data_reg1=combined_data.copy()


## Conduct most basic linear regression on whole period 
#then include time and fixed effects

# Create lagged term of Y1
#data_reg1['Y1_lagged'] = data_reg1[[i for i in range(10)]].shift(1).to_numpy(dtype=float)[:, 0]

# Drop the first row to avoid NaN values from the lag
data_reg1.dropna(inplace=True)

# normalize all time series by dividing by their maximal value
data_reg1['France_News_Index']=data_reg1['France_News_Index'].div(data_reg1['France_News_Index'].max(axis=0), axis=0)

#Choose period:
data_reg1=data_reg1[data_reg1.index> np.datetime64("2014-01-03")]

# Filter the dataset to include only Wednesdays
data_reg1 = data_reg1[data_reg1.index.weekday == 2]


print(data_reg1.head())

import statsmodels.api as sm


def simple_regression(data, factor: int):

    # Define X1 and Y1 with the lagged term included
    X1 = data_reg1[["Deposit-rate", "France_News_Index"]].to_numpy(dtype=float)
    Y1 = data_reg1[factor].to_numpy(dtype=float)

    # Fit the model using statsmodels to get t-values
    X1 = sm.add_constant(X1)  # Adds a constant term to the predictor
    model = sm.OLS(Y1, X1).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    print(f"#######################  Regression for factor {factor+1}: #######################")
    print(model.summary())

# for i in [0,4,9]:
#     simple_regression(data_reg1,i)

def plot_data_reg(data):
    # Plot the data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot factors in the first subplot
    ax1.plot(data.index, data[[0, 4, 9]])
    ax1.set_ylabel('Factors')
    ax1.legend(['Factor 1', 'Factor 5', 'Factor 10'])

    # Plot deposit rate in the second subplot
    ax2.plot(data.index, data["Deposit-rate"], color='tab:orange')
    ax2.set_ylabel('Deposit-rate')
    ax2.legend(['Deposit-rate'])

    # Plot France News Index in the third subplot
    ax3.plot(data.index, data["France_News_Index"], color='tab:green')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('France News Index')
    ax3.legend(['France News Index'])

    plt.tight_layout()
    plt.show()


# data_to_plot=data_reg1[[0,4,9,"Deposit-rate", "France_News_Index"]]
# plot_data_reg(data_to_plot)









