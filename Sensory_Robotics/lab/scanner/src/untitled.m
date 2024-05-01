% Initialisation des variables
num_files = 8; % Nombre de fichiers de données à fusionner
all_points = []; % Initialiser une matrice vide pour stocker tous les points
all_angles = []; % Initialiser une matrice vide pour stocker tous les angles

% Parcourir tous les fichiers de données
for i = 1:num_files
    % Charger les données à partir du fichier .mat
    filename = sprintf('measurement_data_case1%d.mat', i);
    data = load(filename);
    
   % Extraire les données d'inclinaison et de mesure
    tilt_angles = data.tilt_angles;
    measurement_data = data.measurement_data;
    
    % Répéter l'angle d'inclinaison pour chaque valeur du lidar
    tilt_angle_repeated = repmat(tilt_angles(2), length(measurement_data), 1);
    
    % Concaténer les données du lidar avec les angles d'inclinaison correspondants
    combined_data = [tilt_angle_repeated', measurement_data];
    
    % Ajouter les données de cette mesure à la matrice globale
    all_points = [all_points; combined_data];
end

% Affichage du modèle 3D
figure;
scatter3(all_points(:,1), all_points(:,2), all_points(:,3), 'b.');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Modèle 3D fusionné à partir des données du lidar');
