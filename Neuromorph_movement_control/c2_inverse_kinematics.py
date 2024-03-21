"""
Severals robotics spaces

Distance vs Displacement
Distance is the magintudde of the travel path
Displacemet is the difference between two position (starting position - end position)

Speed vs Velocity
Speed is a scalar and valocity is a vector !!!
Velocity is the time rate of change of position
Speed is the magnitude of velocity (rate of change of distance covered by the moving point)

The sequence of rotational displacement of a point iis commutative in 2D but not in 3D space

Inverse kinematics

""" 

import numpy as np
import matplotlib.pyplot as plt

"""
Ce programme à pour but de simuler le comportement d'un bras humain dans 
"""

def inverse_kinematics(init_angles, target, seg_lens, ts_num):
    assert np.sqrt(target[0]**2 + target[1]**2) <= np.sum(seg_lens), "Target point out of reach"

    # Crée une matrice vide de n lignes (correspondant au nombre d'angle) et m colonne (correspondant au nombre de pas)
    alphas = np.zeros((3, ts_num)) 

    # Insère la position initiale de chaque joint dans la premiere colonne de la matrice
    alphas[:, 0] = init_angles

    # Récupère les angles initiaux et les places dans des varibles plus lisibles
    a1 = alphas[0, 0]
    a12 = a1 + alphas[1, 0]
    a123 = a12 + alphas[2, 0]

    # Crée et initialise les matrices du MGD (matrices de taille 2 lignes et ts_num colonne correspondant au nombre de pas à réaliser)
    E1 = np.zeros((2, ts_num))
    E2 = np.zeros((2, ts_num))
    E3 = np.zeros((2, ts_num))

    E1[:, 0] = [seg_lens[0] * np.cos(a1), seg_lens[0] * np.sin(a1)]
    E2[:, 0] = [E1[0, 0] + seg_lens[1] * np.cos(a12), E1[1, 0] + seg_lens[1] * np.sin(a12)]
    E3[:, 0] = [E2[0, 0] + seg_lens[2] * np.cos(a123), E2[1, 0] + seg_lens[2] * np.sin(a123)]

    # Target différence -> (T-EP_0)/1000=[ΔEP_x ΔEP_y]
    target_difference = [(target[0] - E3[0, 0]) / ts_num, (target[1] - E3[1, 0]) / ts_num]

    # Boucle pour la formule J^T \DeltaEP = \Delta\alpha
    for i in range(ts_num - 1):
        
        # Met a jour les variables interne
        a1 = alphas[0, i]
        a12 = a1 + alphas[1, i]
        a123 = a12 + alphas[2, i]

        # Calcule la Jaconbienne du bras
        J = np.array([[-seg_lens[0] * np.sin(a1) - seg_lens[1] * np.sin(a12) - seg_lens[2] * np.sin(a123),
                        -seg_lens[1] * np.sin(a12) - seg_lens[2] * np.sin(a123),
                        -seg_lens[2] * np.sin(a123)],
                       [seg_lens[0] * np.cos(a1) + seg_lens[1] * np.cos(a12) + seg_lens[2] * np.cos(a123),
                        seg_lens[1] * np.cos(a12) + seg_lens[2] * np.cos(a123),
                        seg_lens[2] * np.cos(a123)]])

        # Détermine la différence de la prochaine position à partir de la Jacobienne et de la différence entre la position actuelle et la position finale
        delta_a = np.linalg.pinv(J) @ target_difference

        # Ajoute cette différence à l'ancienne position pour avoir la nouvelle position
        alphas[:, i + 1] = alphas[:, i] + delta_a

    # Place la dernière valeur de la position de chaque joint dans une liste (pour la renvoyer à la fin de la fonction)
    final_a = alphas[:, ts_num - 1]

    # Récupère les valeurs des angles dans une variable plus lisible
    va1 = alphas[0, :]
    va12 = va1 + alphas[1, :]
    va123 = va12 + alphas[2, :]

    E1 = np.array([seg_lens[0] * np.cos(va1), seg_lens[0] * np.sin(va1)])
    E2 = np.array([E1[0, :] + seg_lens[1] * np.cos(va12), E1[1, :] + seg_lens[1] * np.sin(va12)])
    E3 = np.array([E2[0, :] + seg_lens[2] * np.cos(va123), E2[1, :] + seg_lens[2] * np.sin(va123)])

    ### NO JERK - SMOOTH MINIMAL PATH

    tau = np.linspace(0, 1, timesteps_num)
    beta = np.zeros((3, timesteps_num))    # angles by optimal jerk

    beta0 = alphas[:, 0] # defined as an input
    betaT = alphas[:,-1] # calculated from inverse kinematics

    beta0 = beta0.reshape((3, 1))  # Reshape to match dimensions with tau
    betaT = betaT.reshape((3, 1))  # Reshape to match dimensions with tau

    beta = beta0 + (beta0 - betaT) * (15 * tau**4 - 6 * tau**5 - 10 * tau**3)  # using vector, ergo dont have to calculate separately

    b1 = beta[0, :]
    b12 = b1 + beta[1, :]
    b123 = b12 + beta[2, :]

    E1_smooth_jerk = np.array([seg_lens[0] * np.cos(b1), seg_lens[0] * np.sin(b1)])
    E2_smooth_jerk = np.array([E1_smooth_jerk[0, :] + seg_lens[1] * np.cos(b12), E1_smooth_jerk[1, :] + seg_lens[1] * np.sin(b12)])
    E3_smooth_jerk = np.array([E2_smooth_jerk[0, :] + seg_lens[2] * np.cos(b123), E2_smooth_jerk[1, :] + seg_lens[2] * np.sin(b123)])

    # Create a figure
    plt.figure()

    # For the given number of steps
    for i in range(timesteps_num):
        
        # Plot the arm and place its "object" in a variable (for easy removal later)
        arm_iter, = plt.plot([0, E1[0, i], E2[0, i], E3[0, i]], 
                             [0, E1[1, i], E2[1, i], E3[1, i]], 'k-')

        arm_jerk, = plt.plot([0, E1_smooth_jerk[0, i], E2_smooth_jerk[0, i], E3_smooth_jerk[0, i]],
                              [0, E1_smooth_jerk[1, i], E2_smooth_jerk[1, i], E3_smooth_jerk[1, i]], 'b-')

        # Plot the target position and place its "object" in a variable (for easy removal later)
        targ_mark, = plt.plot(target[0], target[1], 'r.', markersize=5)

        # Set the graph limits
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend(['iterative', 'optimal jerk', 'target'])

        # Draw the plot
        plt.draw()

        # Pause for each step depending on the number of steps itself
        plt.pause(0.001 + 100 / (20 * timesteps_num))

        # Remove what has been plotted (the arm and the target position)
        arm_iter.remove()
        arm_jerk.remove()
        targ_mark.remove()

    # Close the simulation window
    plt.close()

    return final_a

segment_lengths = [0.3, 0.25, 0.1] # [Shoulder, forearm, hand]
timesteps_num = 100 # Nombre de pas pour arriver a la target position (précision)
angles = [-np.pi / 2, np.pi / 4, np.pi / 2] # Les 3 angles initiaux des 3 joints

angles = inverse_kinematics(angles, [0.1, 0.3], segment_lengths, timesteps_num)
angles = inverse_kinematics(angles, [0.5, 0.4], segment_lengths, timesteps_num)
angles = inverse_kinematics(angles, [0.3, -0.4], segment_lengths, timesteps_num)
