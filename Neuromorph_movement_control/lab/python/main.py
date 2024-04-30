# import numpy as np

# class Robot:
#     def __init__(self, angle, mechanics_properties):
#         self.alpha = angle
#         self.l = mechanics_properties

#     def EP_x(self):
#         for k in range(len(self.l)):
#             for j in range(k+1):
#                 inner_sum = inner_sum + self.alpha[j]
#             result = result+ (self.l[k] * np.cos(inner_sum))
#         return result

#     def EP_y(self):
#         for k in range(len(self.l)):
#             for j in range(k+1):
#                 inner_sum = inner_sum + self.alpha[j]
#             result = result+ (self.l[k] * np.sin(inner_sum))
#         return result

#     def EP_vx(self):
#         for k in range(len(self.l)):
#             for j in range(k+1):
#                 inner_sum = inner_sum + self.alpha[j]
#             result = result+ (self.l[k] * np.sin(inner_sum))
#         return result

#     def EP_vy(self):
#         for k in range(len(self.l)):
#             for j in range(k+1):
#                 inner_sum = inner_sum + self.alpha[j]
#             result = result+ (self.l[k] * np.sin(inner_sum))
#         return result



# r=Robot([0.3,   # Upper arm
#          0.25,  # Lower arm
#          0.1],  # Hand
#          [2,    # alpha_1 = t
#           5,    # alpha_2 = 2t
#           6])   # alpha_3 = 3t^2



import numpy as np
import matplotlib.pyplot as plt

# Définition des valeurs de L et des fonctions lambda
L1 = 0.3
L2 = 0.25
L3 = 0.1
a1 = lambda t: t
a2 = lambda t: 2*t
a3 = lambda t: 3*t**2

# Création d'un tableau de temps
t_values = np.linspace(1, 300, 3)

# Calcul des positions E1, E2 et E3 pour chaque instant de temps
E1_positions = np.array([[L1*np.cos(a1(t)), L1*np.sin(a1(t))] for t in t_values])
E2_positions = np.array([[E1_positions[0:]+L2*np.cos(a1(t) + a2(t)), E1_positions[1:]+L2*np.sin(a1(t) + a2(t))] for t in t_values])
E3_positions = np.array([[E2_positions[0:]+L3*np.cos(a1(t) + a2(t) + a3(t)), E2_positions[1:]+L3*np.sin(a1(t) + a2(t) + a3(t))] for t in t_values])

for i in range (300):
    plt.plot([0,E1_positions[1:i],E2_positions[1:i],E3_positions[1:i]]
             [0,E1_positions[2:i],E2_positions[2:i],E3_positions[2:i]])
    plt.axis(-0.5,0.5,-0.5,0.5)
    plt.xlim([-1,1])
    plt.xlim([-1,1])
    plt.show()
    plt.pause(0.001)