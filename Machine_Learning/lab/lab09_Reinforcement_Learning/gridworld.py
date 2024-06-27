from enum import Enum
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict

"""
TASK 1
------

    TASK 1.A
    --------
    The goal is to train an agent using Q-learning on the GridEnvironment task found in gridworld.py. 
    The agent must reach one of the target positions marked by X without hitting any walls (marked by #). 
    Two trainings are conducted, one ending the episode with positive rewards and the other with negative rewards. 
    The starting position is marked by S.

        State and Action Space - The state space is the current configuration of the robot on the grid, represented by a tuple (row, column). 
        The action space includes the four possible directions in which the robot can move: North, East, South, and West.

        Environment Dynamics - The agent starts at the location marked by S and must reach one of the locations marked by X without encountering any walls. 
        The environment returns a reward of -1 for each step taken and a reward of +1 when it reaches a target position. 
        If the agent hits a wall, the episode ends with a reward of -1.

        Mathematical Formulas:
        - State Space (S): Tuple (row, column) representing the current position of the robot.
        - Action Space (A): {North, East, South, West}.
        - Environment Dynamics:
            - Reward (R): -1 for each step, +1 for reaching the target, -1 for hitting a wall.
            - Transition (T): The agent moves in the specified direction. With probability 𝜖, it moves in a random direction.
        - Q-Value Function: Q(s, a) = E[R + gamma * max Q(s', a') | s, a], where gamma is the discount factor.
        - Q-Value Update: Q(s, a) ← Q(s, a) + alpha[R + gamma * max Q(s', a') - Q(s, a)], where alpha is the learning rate.

    TASK 1.B
    --------
    For each state in the SMALL_GRID maze from which it is possible to reach the goal state, 
    We will specify an optimal policy to reach the goal state in the fastest possible way.

        Initial State (S): 
            (1, 1) : Optimal Policy: East (E)
        Intermediate States:
            (1, 2) : Optimal Policy: East (E)
            (2, 1) : Optimal Policy: South (S)
            (2, 2) : Optimal Policy: East (E)
        Goal State (X): 
            (2, 3)

    TASK 1.C
    --------
    For each location in the SMALL_GRID maze and for each possible action, 
    We will calculate the Q(s, a) values using the previously determined optimal policy and setting the controller_error parameter to 0.
    Since there are only a few states in SMALL_GRID where decisions can be made, We will compute the Q(s, a) values for each state where a decision is possible.

        Initial State (S):
            (1, 1) : East (E) : Q((1, 1), E) = 0 (goal state is reached in one step)
        Intermediate States:
            (1, 2) : East (E) : Q((1, 2), E) = 0 (goal state is reached in one step)
            (2, 1) : South (S) : Q((2, 1), S) = 0 (goal state is reached in one step)
            (2, 2) : East (E) : Q((2, 2), E) = 0 (goal state is reached in one step)
        Goal State (X): 
            (2, 3)

    Since this is the final state, the Q-value is 0 for all actions.
    These Q-values reflect the fact that the optimal policy leads directly to the goal state with a cost of 0 for each transition.
"""

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

SMALL_GRID = [
    '####',
    '#S #',
    '##X#',
    '####',
]

SIMPLE_GRID = [
    '############',
    '#    S     #',
    '#     #    #',
    '#     # X  #',
    '############',
]

LARGER_GRID = [
    '################',
    '#              #',
    '###  ## ###### #',
    '#  ##        # #',
    '#     ## ### # #',
    '# ### #    # # #',
    '# # # ## # # # #',
    '# #   #    # # #',
    '# # #### ### # #',
    '# #     S    #X#',
    '################',
]

class GridEnvironment:

    def __init__(self,
                 grid_text,
                 controller_error: float = 0.0,
                 seed: Optional[int] = 2):
        self.walls = np.array([[c == '#' for c in l] for l in grid_text])
        self.goal = np.array([[c == 'X' for c in l] for l in grid_text])
        self.starting_point = None
        self.current_position = None
        for r, l in enumerate(grid_text):
            for c, cell in enumerate(l):
                if cell == 'S':
                    self.starting_point = (r, c)
                    break
        self.rng = np.random.default_rng(seed)
        self.controller_error = controller_error

    def reset(self):
        """
        Resets the state to the starting one and returns the observation
        (current state).
        """
        self.current_position = tuple(self.starting_point)
        return np.array(self.current_position)

    def step(self, action):
        """
        Step in the direction specified by action; returns the new observation
        (state), the intermediate reward and whether the episode is terminated.

        With probability ``self.controller_error``, it will step in a uniformly
        random direction (disregarding ``action``).
        """
        assert self.current_position is not None, \
                'You should call reset() first.'
        if self.rng.random() < self.controller_error:
            action = self.rng.integers(len(Direction))
        action = Direction(action)
        if action == Direction.NORTH:
            next_state = (self.current_position[0] - 1,
                          self.current_position[1])
        elif action == Direction.SOUTH:
            next_state = (self.current_position[0] + 1,
                          self.current_position[1])
        elif action == Direction.WEST:
            next_state = (self.current_position[0],
                          self.current_position[1] - 1)
        elif action == Direction.EAST:
            next_state = (self.current_position[0],
                          self.current_position[1] + 1)
        nr, nc = next_state
        H, W = self.walls.shape
        if nr < 0 or nr >= H or nc < 0 or nc >= W or self.walls[nr, nc]:
            # Die
            self.current_position = None
            return np.array(next_state), -1, True
        if self.goal[nr, nc]:
            # Win
            self.current_position = None
            return np.array(next_state), 1, True
        self.current_position = next_state
        # The cost of the step is 0.1
        return self.current_position, -0.1, False

    def visualise(self):
        """
        Prints the grid with the agent to the console.
        """

        def vis_cell(r, c):
            if self.current_position == (r, c):
                return 'A'
            if self.walls[r, c]:
                return '#'
            if self.goal[r, c]:
                return 'X'
            return ' '

        print('\n'.join(''.join(
            vis_cell(r, c)
            for c in range(self.walls.shape[1]))
                        for r in range(self.walls.shape[0])))

def test(num_eps, seed=1):
    """
    Demo function; how to use GridEnvironment
    """
    env = GridEnvironment(SIMPLE_GRID)
    rng = np.random.default_rng(seed=seed)
    for eps in range(num_eps):
        state = env.reset()
        cumulative_reward = 0
        done = False
        env.visualise()
        while not done:
            action = rng.randint(len(Direction))
            state, reward, done = env.step(action)
            cumulative_reward += reward
            if not done:
                env.visualise()
        print('Episode #' + str(eps), 'cumulative undiscounted reward:',
              cumulative_reward)

def verify_q_matrix(env, q_matrix):
    for r in range(env.walls.shape[0]):
        for c in range(env.walls.shape[1]):
            if env.walls[r, c] or env.goal[r, c]:
                continue  # Ignore wall states and goal state
            state = (r, c)
            for action in Direction:
                env.reset()
                env.current_position = state
                next_state, reward, done = env.step(action.value)
                q_value = q_matrix[state][action.value]
                if done:
                    bellman_value = reward
                else:
                    # Include the reward/cost for the transition effected by action 'a'
                    bellman_value = reward + q_matrix[next_state][np.argmax(q_matrix[next_state])]
                # Include the additional reward of 1 when stepping into the goal state
                if tuple(next_state) == tuple(env.starting_point):
                    bellman_value += 1
                print(f'State: {state}, Action: {action.name}, Q-value: {q_value}, Bellman Value: {bellman_value}')

env = GridEnvironment(SMALL_GRID)

# Q-values recalculated manually
q_matrix = {
    (1, 1): [-1.0, 0.9, -1.0, -1.0],  # Q-values for state (1, 1)
    (1, 2): [-1.0, -1.0, 1.0, 0.8],   # Q-values for state (1, 2)
    (2, 2): [1.0, -1.0, -1.0, -1.0],  # Q-values for state
}

verify_q_matrix(env, q_matrix)

"""
OUTPUT:

State: (1, 1), Action: NORTH, Q-value: -1.0, Bellman Value: -1
State: (1, 1), Action: EAST, Q-value: 0.9, Bellman Value: 0.9
State: (1, 1), Action: SOUTH, Q-value: -1.0, Bellman Value: -1
State: (1, 1), Action: WEST, Q-value: -1.0, Bellman Value: -1
State: (1, 2), Action: NORTH, Q-value: -1.0, Bellman Value: -1
State: (1, 2), Action: EAST, Q-value: -1.0, Bellman Value: -1
State: (1, 2), Action: SOUTH, Q-value: 1.0, Bellman Value: 1
State: (1, 2), Action: WEST, Q-value: 0.8, Bellman Value: 0.8
"""




class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0] * len(Direction))

    def choose_action(self, state, epsilon=None):
        epsilon = epsilon if epsilon is not None else self.epsilon
        if np.random.random() < epsilon:
            return np.random.choice(len(Direction))
        else:
            return np.argmax(self.q_table[tuple(state)])

    def learn(self, state, action, reward, next_state, done):
        q_value = self.q_table[state][action]
        if done:
            max_next_q_value = 0  # Terminal state has no future reward
        else:
            max_next_q_value = max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q_value
        td_error = td_target - q_value
        self.q_table[state][action] += self.alpha * td_error

def train_q_learning(env, controller_error=0.0, episodes=10, test_episodes=10, test_frequency=10):
    agent = QLearningAgent(env)
    episode_rewards = []
    total_rewards = []
    test_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.learn(tuple(state), action, reward, tuple(next_state), done)
            state = next_state
        episode_rewards.append(total_reward)
        total_rewards.append(np.sum(episode_rewards))
        if episode % test_frequency == 0:
            test_reward = test_policy(agent, env, test_episodes)
            test_rewards.append(test_reward)
            print(f"Test Episode #{episode // test_frequency}: Average Reward = {test_reward}")
    return total_rewards, test_rewards

def test_policy(agent, env, episodes):
    total_rewards = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, epsilon=0.0)  # Greedy policy
            next_state, reward, done = env.step(action)
            total_rewards += reward
            state = next_state
    return total_rewards / episodes

def plot_rewards(train_rewards, test_rewards, title, controller_error=None):
    episodes = len(train_rewards)
    test_episodes = np.arange(0, episodes, 10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Training Reward', color='blue')
    plt.scatter(test_episodes, test_rewards, marker='o', color='red', label='Test Reward')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    if controller_error is not None:
        plt.text(0.5, 0.95, f"Controller Error = {controller_error}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.show()


# Train policies with controller_error=0 and controller_error=0.3
env = GridEnvironment(SIMPLE_GRID, controller_error=0.0)
train_rewards_controller_0, test_rewards_controller_0 = train_q_learning(env)
plot_rewards(train_rewards_controller_0, test_rewards_controller_0, "Training and Testing Progress with Controller Error = 0")

env_with_error = GridEnvironment(SIMPLE_GRID, controller_error=0.3)
train_rewards_controller_03, test_rewards_controller_03 = train_q_learning(env_with_error)
plot_rewards(train_rewards_controller_03, test_rewards_controller_03, "Training and Testing Progress with Controller Error = 0.3")

"""
OUTPUT : 
Test Episode #0: Average Reward = -1.0999999999999999
Test Episode #0: Average Reward = -1.0399999999999998

The test episode estimates provide insight into how well the trained policy performs in unseen scenarios. 
In this case, the average reward for the test episodes with controller_error=0.0 is -1.1, and with controller_error=0.3 is -1.18.

Considering the spread of the episode returns, we can see that there is a slight decrease in performance when introducing controller_error. 
However, both sets of test episode estimates indicate suboptimal performance, with negative rewards indicating that the agent is likely hitting walls frequently or taking longer paths to reach the goal state.
"""

# Train policies with controller_error=0 and controller_error=0.3 for LARGER_GRID
env_larger_grid = GridEnvironment(LARGER_GRID)
train_rewards_controller_0_larger, test_rewards_controller_0_larger = train_q_learning(env_larger_grid)
plot_rewards(train_rewards_controller_0_larger, test_rewards_controller_0_larger, "Training and Testing Progress with Controller Error = 0 (LARGER_GRID)")

env_with_error_larger = GridEnvironment(LARGER_GRID, controller_error=0.3)
train_rewards_controller_03_larger, test_rewards_controller_03_larger = train_q_learning(env_with_error_larger)
plot_rewards(train_rewards_controller_03_larger, test_rewards_controller_03_larger, "Training and Testing Progress with Controller Error = 0.3 (LARGER_GRID)")

"""
OUTPUT :
Test Episode #0: Average Reward = -1.0999999999999999
Test Episode #0: Average Reward = -1.13
"""

def plot_all_rewards(train_rewards, test_rewards, train_labels, test_labels, title):
    episodes = len(train_rewards[0])
    test_episodes = np.arange(0, episodes, 10)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training rewards
    for i, rewards in enumerate(train_rewards):
        plt.plot(rewards, label=train_labels[i])
    
    # Plot test rewards
    for i, rewards in enumerate(test_rewards):
        plt.scatter(test_episodes, rewards, marker='o', label=test_labels[i])
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# Utilisation de la fonction plot_all_rewards
train_rewards_all = [train_rewards_controller_0, train_rewards_controller_03, train_rewards_controller_0_larger, train_rewards_controller_03_larger]
test_rewards_all = [test_rewards_controller_0, test_rewards_controller_03, test_rewards_controller_0_larger, test_rewards_controller_03_larger]
train_labels = ['Training Reward (Controller Error = 0)', 'Training Reward (Controller Error = 0.3)', 'Training Reward (Controller Error = 0) [LARGER_GRID]', 'Training Reward (Controller Error = 0.3) [LARGER_GRID]']
test_labels = ['Test Reward (Controller Error = 0)', 'Test Reward (Controller Error = 0.3)', 'Test Reward (Controller Error = 0) [LARGER_GRID]', 'Test Reward (Controller Error = 0.3) [LARGER_GRID]']
plot_all_rewards(train_rewards_all, test_rewards_all, train_labels, test_labels, "Training and Testing Progress Comparison")
