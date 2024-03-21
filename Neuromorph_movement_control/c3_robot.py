import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, segment_lengths, init_angles, timesteps_num):
        self.segment_lengths = segment_lengths
        self.init_angles = init_angles
        self.timesteps_num = timesteps_num

    def inverse_kinematics(self, target, minimal_jerk=False):
        assert np.sqrt(target[0]**2 + target[1]**2) <= np.sum(self.segment_lengths), "Target point out of reach"

        # Create an empty matrix with n rows (corresponding to the number of angles) and m columns (corresponding to the number of steps)
        alphas = np.zeros((3, self.timesteps_num)) 

        # Insert the initial position of each joint into the first column of the matrix
        alphas[:, 0] = init_angles

        # Get the initial angles and place them in more readable variables
        a1 = alphas[0, 0]
        a12 = a1 + alphas[1, 0]
        a123 = a12 + alphas[2, 0]

        # Create and initialize matrices of end effector positions (size: 2 rows x self.timesteps_num columns corresponding to the number of steps)
        E1 = np.zeros((2, self.timesteps_num))
        E2 = np.zeros((2, self.timesteps_num))
        E3 = np.zeros((2, self.timesteps_num))

        E1[:, 0] = [self.segment_lengths[0] * np.cos(a1), self.segment_lengths[0] * np.sin(a1)]
        E2[:, 0] = [E1[0, 0] + self.segment_lengths[1] * np.cos(a12), E1[1, 0] + self.segment_lengths[1] * np.sin(a12)]
        E3[:, 0] = [E2[0, 0] + self.segment_lengths[2] * np.cos(a123), E2[1, 0] + self.segment_lengths[2] * np.sin(a123)]

        # Calculate target difference -> (T-EP_0)/self.timesteps_num = [ΔEP_x, ΔEP_y]
        target_difference = [(target[0] - E3[0, 0]) / self.timesteps_num, (target[1] - E3[1, 0]) / self.timesteps_num]

        # Loop for the J^T ΔEP = Δα formula
        for i in range(self.timesteps_num - 1):
            
            # Update internal variables
            a1 = alphas[0, i]
            a12 = a1 + alphas[1, i]
            a123 = a12 + alphas[2, i]

            # Determine the difference in the next position from the Jacobian and the difference between the current position and the target position
            delta_a = np.linalg.pinv(self._jacobian(a1,a12,a123)) @ target_difference

            # Add this difference to the old position to get the new position
            alphas[:, i + 1] = alphas[:, i] + delta_a

        # Place the last value of the position of each joint in a list (to return it at the end of the function)
        final_angles = alphas[:, self.timesteps_num - 1]

        # Get the angle values in a more readable variable
        va1 = alphas[0, :]
        va12 = va1 + alphas[1, :]
        va123 = va12 + alphas[2, :]

        # Calculate the end effector positions based on the final angles
        E1 = np.array([self.segment_lengths[0] * np.cos(va1), self.segment_lengths[0] * np.sin(va1)])
        E2 = np.array([E1[0, :] + self.segment_lengths[1] * np.cos(va12), E1[1, :] + self.segment_lengths[1] * np.sin(va12)])
        E3 = np.array([E2[0, :] + self.segment_lengths[2] * np.cos(va123), E2[1, :] + self.segment_lengths[2] * np.sin(va123)])

        if minimal_jerk==True: E1,E2,E3 = self._minimal_jerk(alphas)

        return E1,E2,E3
    
    def _jacobian(self,a1,a12,a123):
        # Calculate the Jacobian of the arm
        return np.array([[-self.segment_lengths[0] * np.sin(a1) - self.segment_lengths[1] * np.sin(a12) - self.segment_lengths[2] * np.sin(a123),
                        -self.segment_lengths[1] * np.sin(a12) - self.segment_lengths[2] * np.sin(a123),
                        -self.segment_lengths[2] * np.sin(a123)],
                    [self.segment_lengths[0] * np.cos(a1) + self.segment_lengths[1] * np.cos(a12) + self.segment_lengths[2] * np.cos(a123),
                        self.segment_lengths[1] * np.cos(a12) + self.segment_lengths[2] * np.cos(a123),
                        self.segment_lengths[2] * np.cos(a123)]])

    def _minimal_jerk(self, alphas):
        # Create a time vector
        tau = np.linspace(0, 1, self.timesteps_num)

        # Initialize the optimal jerk angles matrix
        beta = np.zeros((3, self.timesteps_num))

        # Initial and final angles (defined and calculated from inverse kinematics)
        beta0 = alphas[:, 0] 
        betaT = alphas[:, -1]

        # Reshape to match dimensions with tau
        beta0 = beta0.reshape((3, 1))
        betaT = betaT.reshape((3, 1))

        # Calculate optimal jerk angles using vectorized operations
        beta = beta0 + (beta0 - betaT) * (15 * tau**4 - 6 * tau**5 - 10 * tau**3)  # using vectorized operations, eliminating the need for separate calculations

        # Extract individual joint angles from the beta matrix
        b1 = beta[0, :]
        b12 = b1 + beta[1, :]
        b123 = b12 + beta[2, :]

        # Calculate end effector positions based on smooth jerk angles
        E1_smooth_jerk = np.array([self.segment_lengths[0] * np.cos(b1), self.segment_lengths[0] * np.sin(b1)])
        E2_smooth_jerk = np.array([E1_smooth_jerk[0, :] + self.segment_lengths[1] * np.cos(b12), E1_smooth_jerk[1, :] + self.segment_lengths[1] * np.sin(b12)])
        E3_smooth_jerk = np.array([E2_smooth_jerk[0, :] + self.segment_lengths[2] * np.cos(b123), E2_smooth_jerk[1, :] + self.segment_lengths[2] * np.sin(b123)])

        return E1_smooth_jerk,E2_smooth_jerk,E3_smooth_jerk


    def plot_arm_movement(self, E1,E2,E3, target):
        # Create a figure
        plt.figure()

        # For the given number of steps
        for i in range(self.timesteps_num):
            
            # Plot the arm and its "object" in a variable for easy removal later
            arm_iter, = plt.plot([0, E1[0, i], E2[0, i], E3[0, i]], 
                                [0, E1[1, i], E2[1, i], E3[1, i]], 'k-')

            # Plot the target position and its "object" in a variable for easy removal later
            targ_mark, = plt.plot(target[0], target[1], 'r.', markersize=5)

            # Set the graph limits
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.legend(['iterative', 'target'])

            # Draw the plot
            plt.draw()

            # Pause for each step depending on the number of steps itself
            plt.pause(0.001 + 100 / (20 * self.timesteps_num))

            # Remove what has been plotted (the arm and the target position)
            arm_iter.remove()
            targ_mark.remove()

        # Close the simulation window
        plt.close()

segment_lengths = [0.3, 0.25, 0.1]  # [Shoulder, forearm, hand]
init_angles = [-np.pi / 2, np.pi / 4, np.pi / 2]  # The initial angles of the joints
timesteps_num = 100  # Number of steps to reach the target position (precision)

# Create a robot instance
robot = Robot(segment_lengths, init_angles, timesteps_num)

# Plot arm movement and target position for each target
for target_position in [[0.1, 0.3], [0.5, 0.4], [0.3, -0.5]]:
    E1,E2,E3 = robot.inverse_kinematics(target_position, minimal_jerk=True)
    robot.plot_arm_movement(E1,E2,E3,target_position)
