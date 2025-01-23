import pandas as pd

data = pd.read_excel("Code\EtudeDonneeDetteGenerale\BDF_Tresor_2004_2008_maturites10_quotidien.xlsx")

for i in [1,2,3,5,7,10,15,20,25,30]:
    print(data(data["Maturite"] == i).mean())