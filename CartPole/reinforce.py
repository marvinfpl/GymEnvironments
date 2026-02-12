import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n 

GAMMA = 0.99
TRAINING = 10000
EVAL = 100
LR = 1e-3
GRAD_NORM = 2.0
COEF_ENTROPY = 0.05

class Reinforce(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, f_out),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    
reinforce = Reinforce(n_states, n_actions)
optimizer = optim.Adam(reinforce.parameters(), lr=LR)

def compute_returns(rewards):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


reinforce.train()
mean_rewards = deque(maxlen=100)
for episode in range(TRAINING):
    state, _ = env.reset()
    total_reward = 0.0
    done = False

    log_probs = []
    rewards = []
    entropies = []
    loss = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = reinforce(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        rewards.append(reward)
        done = terminated or truncated

    mean_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(mean_rewards)} ---")

    returns = compute_returns(rewards)
    log_probs = torch.stack(log_probs)
    entropy = torch.stack(entropies).mean()

    loss = -(log_probs * returns).mean() - COEF_ENTROPY * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(reinforce.parameters(), GRAD_NORM)
    optimizer.step()

reinforce.eval()
for episode in range(EVAL):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = reinforce(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f" --- Episode: {episode}, Total Reward: {total_reward} --- ")

env.close()