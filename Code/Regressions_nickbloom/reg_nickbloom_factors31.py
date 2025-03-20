#part des facteurs estimé et d'une interpolation de l'indice de Nick Bloom. Fait la régression.
#Utilise des contrôles éventuels

import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import os



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
data_ECB.columns=['Date',"Time_period","Deposit-rate"]
data_ECB['Date'] = pd.to_datetime(data_ECB['Date'])

### Process données en 1 dataframe


### Merge data_ECB and data_NB with data_factors based on the closest date

#set dates to indexes
data_NB.set_index('Date', inplace=True)
data_factors.set_index('Date', inplace = True)
data_ECB.set_index('Date', inplace=True)



# print(data_ECB.head())
# print(data_factors.head())
# print(data_NB.head())


# Interpolate data_ECB and data_NB to match the dates in data_factors
data_ECB = data_ECB.reindex(data_ECB.index)

data_NB = data_NB.reindex(data_ECB.index)

# Concatenate the datasets
combined_data = pd.concat([data_factors, data_ECB['Deposit-rate'], data_NB], axis=1)
combined_data=combined_data[combined_data.index> np.datetime64("2006-12-31")]


#Interpolate all values on a daily basis
combined_data.interpolate(inplace=True)    

combined_data=combined_data[combined_data.index >np.datetime64("2007-01-02")]



# Print the combined data to verify
print(combined_data.head())

######################### 1st regression: monthly basis ############################################

## Conduct most basic linear regression on whole period 
#then include time and fixed effects

X1 = combined_data[["Deposit-rate","France_News_Index"]].to_numpy(dtype=float)
Y1 = combined_data[[i for i in range(10)]].to_numpy(dtype=float)

# reg1=LinearRegression.fit(X1,Y1)

import statsmodels.api as sm

# Fit the model using statsmodels to get t-values
X1 = sm.add_constant(X1)  # Adds a constant term to the predictor
model = sm.OLS(Y1, X1).fit()

# Summarize the results of the regression
print("Coefficients:", model.params)
print("Intercept:", model.params[0])
print("R^2 score:", model.rsquared)
print("t-values:", model.tvalues)
print("p-values:", model.pvalues)




############# add a lag ? ##################




