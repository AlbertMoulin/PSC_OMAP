import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

dico = {'1er octobre 2024': '2024-10-01', '8 octobre 2024': '2024-10-08', '15 octobre 2024': '2024-10-15', '22 octobre 2024': '2024-10-22', '29 octobre 2024': '2024-10-29', '5 novembre 2024': '2024-11-05', '12 novembre 2024': '2024-11-12', '19 novembre 2024': '2024-11-19', '26 novembre 2024': '2024-11-26', '3 décembre 2024': '2024-12-03', '10 décembre 2024': '2024-12-10', '17 décembre 2024': '2024-12-17', '24 décembre 2024': '2024-12-24', '31 décembre 2024': '2024-12-31', '7 janvier 2025': '2025-01-07', '14 janvier 2025': '2025-01-14', '21 janvier 2025': '2025-01-21', '28 janvier 2025': '2025-01-28', '4 février 2025': '2025-02-04', '11 février 2025': '2025-02-11', '18 février 2025': '2025-02-18', '25 février 2025': '2025-02-25', '4 mars 2025': '2025-03-04', '11 mars 2025': '2025-03-11', '18 mars 2025': '2025-03-18', '25 mars 2025': '2025-03-25', '1er avril 2025': '2025-04-01', '8 avril 2025': '2025-04-08', '15 avril 2025': '2025-04-15'}

# importation des données
file_path = "../data_QAG/donnees_fusionnees_XV.csv" 
df_XV = pd.read_csv(file_path)

df_XV.rename(columns={'question.textesQuestion.texteQuestion.infoJO.dateJO': 'date', 'question.textesQuestion.texteQuestion.texte' : 'texteQuestion'}, inplace=True)

relevant_columns = ['date', 'texteQuestion']
df_XV = df_XV[relevant_columns]
df_XV['date'] = pd.to_datetime(df_XV['date'], errors='coerce', dayfirst=True)
print("finished reading df_XV")

file_pathXIV = '../data_QAG/donnees_fusionnees_XIV.csv'
df_XIV = pd.read_csv(file_pathXIV)
df_XIV.rename(columns={'textesReponse.texteReponse.infoJO.dateJO': 'date', 'textesReponse.texteReponse.texte' : 'texteQuestion'}, inplace=True)
df_XIV = df_XIV[relevant_columns]
df_XIV['date'] = pd.to_datetime(df_XIV['date'], errors='coerce', dayfirst=True)
print("finished reading df_XIV")

file_pathXVI = '../data_QAG/donnees_fusionnees_XVI.csv'
df_XVI = pd.read_csv(file_pathXVI)
df_XVI.rename(columns={'question.minAttribs.minAttrib.infoJO.dateJO': 'date', 'question.textesReponse.texteReponse.texte' : 'texteQuestion'}, inplace=True)
print("finished reading df_XVI")

file_pathXVII = '../data_QAG/donnees_fusionnees_XVII_test.csv'
def date(date_str):
    return dico[date_str]
    
df_XVII = pd.read_csv(file_pathXVII, on_bad_lines='skip')

df_XVII.rename(columns={'date_question': 'date', 'question_text' : 'texteQuestion'}, inplace=True)
df_XVII['date'] = df_XVII['date'].apply(date)
df_XVII['date'] = pd.to_datetime(df_XVII['date'], errors='coerce')
print("finished reading df_XVII")


#recherche d'apparition des mots clés
def contains_keywords(text, keywords_dette, keywords_incertitude, keywords_économie):
    if text is None or pd.isna(text):
        return False
    text = text.lower()
    found_dette = any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords_dette)
    found_incertitude = any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords_incertitude)
    found_économie = any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords_économie)
    return found_dette and found_incertitude and found_économie


def dailycount(df, keywords_dette, keywords_incertitude, keywords_économie):
    df['keywords_present'] = df['texteQuestion'].apply(contains_keywords, keywords_dette=keywords_dette, keywords_incertitude=keywords_incertitude, keywords_économie=keywords_économie)
    daily_counts = df.groupby('date')['keywords_present'].agg(['sum', 'size'])
    #daily_counts['keywords_frequency'] = daily_counts['sum'] / daily_counts['size']
    daily_counts['keywords_frequency'] = daily_counts['sum']
    daily_counts.rename(columns={'sum': 'total_keywords_count', 'size': 'question_count'}, inplace=True)
    daily_counts.reset_index(inplace=True)
    return daily_counts

def display_frequency_week(df, keywords):
    daily_counts = dailycount(df, keywords)
    print("here are the colums", daily_counts.columns)
    daily_counts.plot(x='date', y='keywords_frequency', kind='line', title='Fréquence des mots-clés par date')
    plt.xlabel('Date')
    plt.ylabel('Fréquence des mots-clés')
    plt.savefig("/Users/julietteanglade/Desktop/X2023/2A/PSC/PSC_OMAP-1/Données_scrapping/graphes/courbe_test_1_XIV.png")
    plt.show()



def display_frequency_month(df, keywords_dette, keywords_incertitude, keywords_économie):
    daily_counts = dailycount(df, keywords_dette=keywords_dette, keywords_incertitude=keywords_incertitude, keywords_économie=keywords_économie)
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts.set_index('date', inplace=True)
    monthly_counts = daily_counts.resample('ME').mean()
    monthly_counts.reset_index(inplace=True)
    monthly_counts.plot(x= 'date', y='keywords_frequency', kind='line', title='Fréquence mensuelle des ')
    start_date = 2012
    end_date = 2025
    years = range(start_date, end_date)

#lignes verticales au moment du début des débats pour la loi de finances
    for year in years:
        date = pd.Timestamp(f"{year}-10-01")
        plt.axvline(x=date, color='gray', linestyle='-', alpha=0.2)
    
    evenements = {
    '2017-05-07': ('Élection Macron', 'orange'),
    '2018-11-17': ('Gilets Jaunes', 'purple'),
    '2020-03-15': ('Début Covid-19', 'blue'),
    '2022-02-24': ('Guerre Ukraine', 'green')
    }
# Ajout des lignes verticales avec légendes
    for date, (evenement, couleur) in evenements.items():
        date = pd.to_datetime(date)
        plt.axvline(x=date, color=couleur, linestyle='--', alpha=0.5, label=evenement)
    
    plt.xlabel('Mois')
    plt.ylabel('Fréquence des mots-clés')
    plt.savefig("../résultats/our_index.png")
    plt.show()

    save_path = "/Users/julietteanglade/Desktop/X2023/2A/PSC/PSC_OMAP-1/Données_scrapping/questions_gouv/instability_index.csv"
    monthly_counts.rename(columns={'keywords_frequency': 'instability_index'}, inplace=True)
    monthly_counts.to_csv(save_path, index=False)

#concaténation des dataframes
def show_all(list_df):
    return pd.concat(list_df)

all_df = show_all([df_XIV,df_XV, df_XVI, df_XVII])


keywords_dette = ['dette', 'déficit', 'taux d\'intérêt', 'inflation', 'crise budgétaire', 'finances publiques', 'budget']
keywords_incertitude = ['incertitude', 'crise', 'instabilité', 'risque', 'menace', 'incertain']
keywords_économie = ['économie', 'croissance', 'PIB', 'marché', 'investissement', 'économique']

display_frequency_month(all_df, keywords_dette, keywords_incertitude, keywords_économie)




