% Charger le fichier .mat
matData = load('../data_dette/nusrates_dette.mat'); 

disp(fieldnames(matData)); % Affiche les noms des champs dans la structure

% % Accéder à la variable (par exemple : data)
data = matData.mdate; 

% % Sauvegarder en fichier .csv
writematrix(data, '../data_dette/mdate11.csv');