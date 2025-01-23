% Liste des fichiers CSV
csvFiles = {'Code\code_papier_calvet_18_12\data_dette_cleaned\mat_cleaned.csv', 'Code\code_papier_calvet_18_12\data_dette_cleaned\mdate_cleaned.csv', 'Code\code_papier_calvet_18_12\data_dette_cleaned\mrate_cleaned.csv', 'Code\code_papier_calvet_18_12\data_dette_cleaned\swapmat.csv'};

% Liste des fichiers CSV
tabnames = {'mat', 'mdate', 'rates','swapmat'};

% Initialisation d'une structure pour stocker les données
data = struct();

% Lire chaque fichier CSV et stocker les données
for i = 1:length(csvFiles)
    fileName = csvFiles{i};
    dataName = [tabnames{i}]; % Exemple : 'table1', 'table2', ...
    data.(dataName) = readmatrix(fileName); % Lecture du fichier CSV
end

% Sauvegarder toutes les données dans un fichier MAT
save('Code\code_papier_calvet_18_12\data_dette_cleaned\nusrates_dette_cleaned.mat', '-struct', 'data');
