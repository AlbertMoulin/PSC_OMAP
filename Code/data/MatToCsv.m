% Charger le fichier .mat
matData = load('/MATLAB Drive/data_dette/nusrates_dette.mat'); 

disp(fieldnames(matData)); % Affiche les noms des champs dans la structure

% Accéder à la variable (par exemple : data)
data = matData.rates; 

% Sauvegarder en fichier .csv
writematrix(data, '/MATLAB Drive/data_dette/ rates.csv');