import pandas as pd

data = pd.read_excel("Code\EtudeDonneeDetteGenerale\BDF_Tresor_2004_2008_maturites10_quotidien.xlsx")

print(data.columns)

# delete data before 2007-01-01
data = data[data['Date'] >= '2007-01-01']

# for j in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
#     print(type(data[j][0]))

# clean the data by converting to float when possible and interpolating when it's not
for i in [1,2,3,5,7,10,15,20,25,30]:
    data[i] = pd.to_numeric(data[i], errors='coerce')
    data[i] = data[i].interpolate()

print(data.head())
print(data.iloc[-1])


for i in [1,2,3,5,7,10,15,20,25,30]:
    print(f'Mean for maturity {i} :')
    print(data[i].mean())

# save the cleaned data
data.to_csv("Code\EtudeDonneeDetteGenerale\BDF_Tresor_2004_2008_maturites10_quotidien_cleaned.csv", index=False)
