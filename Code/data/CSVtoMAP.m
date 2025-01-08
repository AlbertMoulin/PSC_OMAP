% Liste des fichiers CSV
csvFiles = {'../data_dette/mat.csv', '../data_dette/Mdates_dette.csv', '../data_dette/rates.csv',  '../data_dette/swapmat.csv'};

% Liste des fichiers CSV
tabnames = {'mat', 'mdate', 'rates', 'swapmat'};

% Initialisation d'une structure pour stocker les données
data = struct();

% Lire chaque fichier CSV et stocker les données
for i = 1:length(csvFiles)
    fileName = csvFiles{i};
    dataName = [tabnames{i}]; % Exemple : 'table1', 'table2', ...
    data.(dataName) = readmatrix(fileName); % Lecture du fichier CSV
end

% Sauvegarder toutes les données dans un fichier MAT
save('../data_dette/nusrates_dette.mat', '-struct', 'data');