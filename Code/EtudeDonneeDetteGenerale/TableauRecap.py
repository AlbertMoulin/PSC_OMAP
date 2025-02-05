import pandas as pd

data = pd.read_excel("Code\EtudeDonneeDetteGenerale\BDF_Tresor_2004_2008_maturites10_quotidien.xlsx")

# delete data before 2007-01-01
data = data[data['Date'] >= '2007-01-01']

# for j in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
#     print(type(data[j][0]))

# clean the data by converting to float when possible and interpolating when it's not
for i in [1,2,3,5,7,10,15,20,25,30]:
    data[i] = pd.to_numeric(data[i], errors='coerce')
    data[i] = data[i].interpolate()


print(data.describe())
print(data.size)
print(data.shape)
print(data.head())

# print(data.head())
# print(data.iloc[-1])


# for i in [1,2,3,5,7,10,15,20,25,30]:
#     print(f'Mean for maturity {i} :')
#     print(data[i].mean())

# calculate std dev for every maturity
# for i in [1,2,3,5,7,10,15,20,25,30]:
#     print(f'Standard deviation for maturity {i} :')
#     print(data[i].std())

# # calculate skewness for every maturity
# for i in [1,2,3,5,7,10,15,20,25,30]:
#     print(f'Skewness for maturity {i} :')
#     print(data[i].skew())

# calculate excess kurtosis for every maturity
# for i in [1,2,3,5,7,10,15,20,25,30]:
#     print(f'Excess kurtosis for maturity {i} :')
#     print(data[i].kurtosis())

# calculate weekly autocorrelation for every maturity of order 1,5,10,20
# for i in [1,2,3,5,7,10,15,20,25,30]:
#     for j in [1,5,10,20]:
#         print(f'Autocorrelation for maturity {i} of order {j} :')
#         print(data[i].autocorr(lag=j))


#convert dates to integer as a serial date number

# # save the cleaned dates without the header
# data['Date'].to_csv("Code\EtudeDonneeDetteGenerale\mdate_cleaned.csv", index=False, header=False)

# # save the cleaned rates without the header
# data[[1,2,3,5,7,10,15,20,25,30]].to_csv("Code\EtudeDonneeDetteGenerale\mrate_cleaned.csv", index=False, header=False)

# # save the maturities in csv files
# maturities = pd.DataFrame([1,2,3,5,7,10,15,20,25,30])
# maturities.to_csv("Code\EtudeDonneeDetteGenerale\mat_cleaned.csv", index=False, header=False)
# maturities.to_csv("Code\EtudeDonneeDetteGenerale\swapmat.csv", index=False, header=False)


