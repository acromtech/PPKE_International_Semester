import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx

# Paramètres de l'environnement
landmarks = np.array([[10, 10], [50, 30], [20, 60]])  # Positions des amers (x, y)
obstacles = np.array([[30, 20], [40, 40], [20, 50]])  # Positions des obstacles (x, y)
start_point = np.array([5, 5])  # Point de départ (x, y)
end_point = np.array([70, 70])  # Point d'arrivée (x, y)

# Créer un graphe pour le plan de l'environnement
G = nx.Graph()

# Ajouter les nœuds (points d'atterrissage, amers, obstacles)
for landmark in landmarks:
    G.add_node(tuple(landmark))
for obstacle in obstacles:
    G.add_node(tuple(obstacle))
G.add_node(tuple(start_point))
G.add_node(tuple(end_point))

# Ajouter les arêtes (connexions entre les nœuds accessibles)
for node1 in G.nodes():
    for node2 in G.nodes():
        if node1 != node2:
            dist = distance.euclidean(node1, node2)
            if dist < 20:  # Arbitrairement choisi pour connecter les nœuds proches
                G.add_edge(node1, node2, weight=dist)

# Trouver le chemin optimal avec l'algorithme A*
path = nx.astar_path(G, tuple(start_point), tuple(end_point))

# Extraire les coordonnées du chemin trouvé
path_x = [point[0] for point in path]
path_y = [point[1] for point in path]

# Affichage du chemin trouvé et de l'environnement
plt.figure()
plt.scatter(path_x, path_y, color='blue', label='Optimal Path')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='o', label='Landmarks')
plt.scatter(obstacles[:, 0], obstacles[:, 1], c='k', marker='x', label='Obstacles')
plt.scatter(start_point[0], start_point[1], c='g', marker='o', label='Start Point')
plt.scatter(end_point[0], end_point[1], c='m', marker='o', label='End Point')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Path Planning')
plt.legend()
plt.grid(True)
plt.show()
