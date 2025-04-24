import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = "C:/code/PSC_OMAP/Données_scrapping/donnees_fusionnees.csv" 
df = pd.read_csv(file_path)

# Print the label names (column names) of the DataFrame
print(df.columns)
# Extract the relevant columns into a new DataFrame
columns_of_interest = ['question.indexationAN.analyses.analyse', 'question.auteur.groupe.abrege', 'question.minAttribs.minAttrib.infoJO.dateJO', 'question.textesQuestion', 'question.textesReponse.texteReponse.texte']
df_filtered = df[columns_of_interest]

#df_filtered['question.minAttribs.minAttrib.infoJO.dateJO'] = pd.to_datetime(df_filtered['question.minAttribs.minAttrib.infoJO.dateJO'], errors='coerce')

#Compte les occurences du mot dette
df_filtered['dette_count'] = df_filtered['question.textesReponse.texteReponse.texte'].str.contains('dette', case=False, na=False).astype(int)

#idem pour taux d'intérêt
df_filtered['taux_interet_count'] = df_filtered['question.textesReponse.texteReponse.texte'].str.contains("taux d'intérêt", case=False, na=False).astype(int)

#regroupé par dates les questions
df_grouped = df_filtered.groupby('question.minAttribs.minAttrib.infoJO.dateJO')[['dette_count', 'taux_interet_count']].sum().reset_index()

#renommer les colonnes:


df_grouped.columns = ['date', 'dette_count', 'taux_interet_count']

print(df_grouped)
import matplotlib.pyplot as plt

# Plot the frequency of occurrences
plt.figure(figsize=(12, 6))
plt.plot(df_grouped['date'], df_grouped['dette_count'], label='Dette Count', marker='o')
plt.plot(df_grouped['date'], df_grouped['taux_interet_count'], label="Taux d'intérêt Count", marker='o')

# Add titles and labels
plt.title('Fréquence des termes "dette" et "taux d intérêt" durant les séances de questions au gouvernement')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()

# Reduce the number of date labels shown
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

plt.show()

