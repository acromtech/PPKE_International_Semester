import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class PolicyModel(nn.Module):
    """
    Model that calculates the action probabilities from the state.

    Returns with the action probabilities and the logarithm of these (for more
    stable calculations).
    """

    def __init__(self):
        super().__init__()
        # yapf: disable
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2))
        # yapf: enable

    def forward(self, x):
        x = self.model(x)
        return (nn.functional.softmax(x, dim=1),
                nn.functional.log_softmax(x, dim=1))

def sample(model, state):
    action_probs, _ = model(state.view(1, -1))
    dist = torch.distributions.Categorical(action_probs)
    return dist.sample().item()

def pg_loss(taken_action, log_pi, cum_reward):
    # TODO
    pass

def train_on_episode(model, optimiser, trajectories, reward_baseline):
    # Create dataset
    # TODO calculate cumulated rewards, etc
    # Optim update
    optimiser.zero_grad()
    _, log_pi = model(dataset_states)
    loss = pg_loss(dataset_actions, log_pi, dataset_cum_rewards)
    loss.backward()
    optimiser.step()
    return loss

def train_epoch(model, optimiser, env, global_rewards):
    # TODO sample trajectories through interaction
    # TODO call `train_on_episode` to update the model

def train(model, num_epochs):
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    env = gym.make('CartPole-v0')
    rewards = []
    for e in range(num_epochs):
        loss = train_epoch(model, optimiser, env, rewards)
        print(f'Epoch {e}:  loss: {loss}')
    env.close()

def evaluate(model, num_eps):
    env = gym.make('CartPole-v0')
    with torch.no_grad():
        for _ in range(num_eps):
            state = env.reset()
            done = False
            T = 0
            while not done:
                env.render()
                state_pt = torch.from_numpy(state.astype(np.float32))
                action = sample(model, state_pt)
                state, reward, done, _ = env.step(action)
                T += 1
            print(f'Episode length: {T}')
    env.close()
