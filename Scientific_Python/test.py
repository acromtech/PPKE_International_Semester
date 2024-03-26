import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'environnement et des amers
landmarks = np.array([[0, 0], [5, 2], [2, 5]])  # Positions des amers (x, y)
num_landmarks = landmarks.shape[0]  # Nombre d'amers
initial_robot_pose = np.array([5, 5, np.pi/4])  # Position et orientation initiales du robot (x, y, theta)

# Paramètres de simulation
num_time_steps = 50  # Nombre d'étapes de temps
motion_noise = 0.1  # Réduire le bruit de mouvement
measurement_noise = 0.1  # Réduire le bruit de mesure
wheel_radius = 0.1  # Rayon des roues
wheel_distance = 0.5  # Distance entre les roues

def generate_motion(robot_pose, left_wheel_speed, right_wheel_speed):
    # Calcul de la vitesse linéaire et angulaire du robot
    linear_velocity = (left_wheel_speed + right_wheel_speed) / 2
    angular_velocity = (right_wheel_speed - left_wheel_speed) / 0.5  # Largeur de l'essieu du robot (distance entre les roues)

    # Calcul des déplacements linéaire et angulaire pendant une étape de temps
    dt = 0.1  # Pas de temps
    dx = linear_velocity * np.cos(robot_pose[2]) * dt
    dy = linear_velocity * np.sin(robot_pose[2]) * dt
    dtheta = angular_velocity * dt

    return np.array([dx, dy, dtheta])

# Générer les mesures des amers
def generate_measurements(robot_pose):
    true_distances = np.linalg.norm(landmarks - robot_pose[:2], axis=1)
    measured_distances = true_distances + np.random.randn(num_landmarks) * measurement_noise
    measured_bearings = np.arctan2(landmarks[:, 1] - robot_pose[1], landmarks[:, 0] - robot_pose[0]) - robot_pose[2]
    return np.vstack((measured_distances, measured_bearings)).T

# Initialiser le filtre de Kalman
def init_kalman_filter():
    # Matrice d'état x contenant les coordonnées du robot (x, y) et son orientation theta
    # Nous initialisons également la matrice de covariance P avec des valeurs arbitraires
    global x, P
    x = np.zeros(3)  # Initialisation de l'état à zéro
    P = np.eye(3)    # Initialisation de la matrice de covariance avec l'identité

# Mise à jour du filtre de Kalman
def update_kalman_filter(control, measurements):
    global x, P
    # Prédiction de l'état suivant en utilisant les équations de mouvement
    x[0] += control[0]
    x[1] += control[1]
    x[2] += control[2]

    # Calcul de la matrice Jacobienne de la fonction de transition
    F = np.array([[1, 0, -control[1] * np.sin(x[2])],
                  [0, 1, control[1] * np.cos(x[2])],
                  [0, 0, 1]])

    # Prédiction de la covariance de l'état suivant
    Q = np.eye(3) * motion_noise**2  # Matrice de covariance du bruit de mouvement
    P = F.dot(P).dot(F.T) + Q

    # Calcul de la matrice Jacobienne de la fonction de mesure
    H = np.zeros((num_landmarks * 2, 3))
    for i in range(num_landmarks):
        delta_x = landmarks[i, 0] - x[0]
        delta_y = landmarks[i, 1] - x[1]
        q = delta_x**2 + delta_y**2
        H[2*i, :] = [-delta_x / np.sqrt(q), -delta_y / np.sqrt(q), 0]
        H[2*i+1, :] = [delta_y / q, -delta_x / q, -1]

    # Calcul de la matrice de covariance du bruit de mesure
    R = np.eye(num_landmarks * 2) * measurement_noise**2

    # Calcul de la résidu de la mesure
    y = measurements - np.vstack((np.linalg.norm(landmarks - x[:2], axis=1),
                                  np.arctan2(landmarks[:, 1] - x[1], landmarks[:, 0] - x[0]) - x[2])).T

    # Correction de l'état prédit en utilisant les mesures
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x += K.dot(y.reshape(-1, 1)).flatten()

    # Correction de la covariance
    P = (np.eye(3) - K.dot(H)).dot(P)

# Simulation de la localisation du robot
# Simulation de la localisation du robot
def simulate_localization():
    robot_pose = initial_robot_pose
    init_kalman_filter()

    # Créer une seule et même fenêtre pour les deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Contrôles (vitesses des roues)
    left_wheel_speed = 0.2  # Vitesse de la roue gauche (avancer plus rapidement)
    right_wheel_speed = 0.2  # Vitesse de la roue droite (avancer plus rapidement)

    for t in range(num_time_steps):
        # Générer les mouvements du robot
        control = generate_motion(robot_pose, left_wheel_speed, right_wheel_speed)

        # Générer les mesures des amers
        measurements = generate_measurements(robot_pose)

        # Mettre à jour le filtre de Kalman avec les nouvelles données
        update_kalman_filter(control, measurements)

        # Dessiner la trajectoire du robot (un cercle)
        circle = plt.Circle((x[0], x[1]), 0.5, color='blue', fill=False)
        ax1.add_artist(circle)

        # Ajouter la position estimée du robot aux listes
        estimated_x.append(x[0])
        estimated_y.append(x[1])

        # Effacer les graphiques précédents
        ax1.clear()
        ax2.clear()

        # Sous-graphique 1 : Trajectoire estimée du robot
        ax1.plot(estimated_x, estimated_y, label='Estimated Robot Path')
        ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='o', label='Landmarks')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Robot Localization')
        ax1.legend()
        ax1.grid(True)

        # Calculer les bornes des axes pour centrer le graphique sur le robot
        min_x = min(estimated_x) - 2
        max_x = max(estimated_x) + 2
        min_y = min(estimated_y) - 2
        max_y = max(estimated_y) + 2

        # Sous-graphique 2 : Zoom sur la trajectoire estimée du robot
        ax2.plot(estimated_x, estimated_y, label='Estimated Robot Path')
        ax2.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='o', label='Landmarks')
        ax2.axis([min_x, max_x, min_y, max_y])  # Ajuster les limites des axes pour centrer le graphique sur le robot
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Robot Localization (Zoom)')
        ax2.legend()
        ax2.grid(True)

        # Mettre à jour le graphique
        plt.pause(0.01)


# Listes pour stocker les coordonnées estimées du robot
estimated_x = []
estimated_y = []

# Appel de la fonction de simulation
simulate_localization()


