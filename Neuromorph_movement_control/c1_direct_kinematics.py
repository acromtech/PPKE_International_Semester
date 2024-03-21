import numpy as np
import matplotlib.pyplot as plt

# Direct kinematic

L1 = 0.3  # Upper arm length
L2 = 0.25  # Lower arm
L3 = 0.1   # Hand
L4 = 0.05  # Pencil

t = np.linspace(0, 3, 300)  # Time 0s -> 3s

a1 = 2 * t       # Shoulder angle
a2 = 3 * t**2    # Elbow angle
a3 = t           # Wrist angle
a4 = t / 2       # Pencil and wrist angle

a12 = a1 + a2          # Calculate the summed angles for the next step
a123 = a12 + a3
a1234 = a123 + a4

# Compute vertex locations
E1 = np.array([L1 * np.cos(a1), L1 * np.sin(a1)])   # Coordinates of the elbow
E2 = np.array([E1[0, :] + L2 * np.cos(a12), E1[1, :] + L2 * np.sin(a12)])   # Coordinates of the wrist
E3 = np.array([E2[0, :] + L3 * np.cos(a123), E2[1, :] + L3 * np.sin(a123)])  # Coordinates of the finger tip
E4 = np.array([E3[0, :] + L4 * np.cos(a1234), E3[1, :] + L4 * np.sin(a1234)])  # Coordinates: end of the pencil

# Plot the arm movements
plt.figure()
for i in range(300):
    plt.clf()
    plt.plot([0, E1[0, i], E2[0, i], E3[0, i], E4[0, i]], [0, E1[1, i], E2[1, i], E3[1, i], E4[1, i]])
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.draw()
    plt.pause(0.001)

plt.show()
