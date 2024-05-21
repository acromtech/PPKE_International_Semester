from enum import Enum
import numpy as np
from typing import Optional

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
                    bellman_value = reward + q_matrix[next_state][np.argmax(q_matrix[next_state])]
                print(f'State: {state}, Action: {action.name}, Q-value: {q_value}, Bellman Value: {bellman_value}')


env = GridEnvironment(SMALL_GRID)

# Q-values recalculated manually
q_matrix = {
    (1, 1): [-1.0, 0.9, -1.0, -1.0],  # Q-values for state (1, 1)
    (1, 2): [-1.0, -1.0, 1.0, 0.8],   # Q-values for state (1, 2)
    (2, 2): [1.0, -1.0, -1.0, -1.0],  # Q-values for state
}

verify_q_matrix(env, q_matrix)