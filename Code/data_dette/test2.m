% Charger le fichier CSV dans une table

load(['../data_dette/nusrates_dette.mat'],'rates','mat','mdate','-mat');
data = rates;

% Vérifier chaque ligne
for index = 1:height(data)
    % Vérifier la longueur de la ligne
    disp(width(data(index, :)))
    if width(data(index, :)) ~= 8
        disp(data(index, :)); % Afficher la ligne si la longueur est différente de 8
    end
end
