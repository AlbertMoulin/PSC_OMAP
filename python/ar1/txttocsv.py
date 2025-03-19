import csv

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as infile, open(csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        for line in infile:
            # Remplace les espaces multiples par un seul espace, puis split
            row = line.strip().split()
            writer.writerow(row)

# Exemple d'utilisation
txt_to_csv('C:/Users/alex/Desktop/psc/ar1/data_x5.txt', 'C:/Users/alex/Desktop/psc/ar1/data_x5.csv')