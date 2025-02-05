import os
import pandas as pd

fichiers_json = 'C:/code/PSC_OMAP/Données_scrapping/json'

try:
    # Example of reading a JSON file
    df = pd.read_json(os.path.join(fichiers_json, 'QANR5L16QG1.json'))
    print(df)
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")