import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

GAMMA = 0.99
TRAINING = 2000
EVAL = 100
LR_ACTOR = 3e-3
LR_CRITIC = 4e-3
COEF_ENTROPY = 0.05
ACTOR_GRAD_NORM = 2.0

env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

class Actor(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, f_out),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, f_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)
    
actor = Actor(n_states, n_actions)
optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic = Critic(n_states)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

def compute_returns(rewards):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    return returns

actor.train()
critic.train()
mean_rewards = deque(maxlen=100)

for episode in range(TRAINING):
    state, _ = env.reset()
    total_reward = 0.0
    done = False

    log_probs = []
    states = []
    rewards = []
    values = []
    entropies = []

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        states.append(state_t)
        entropies.append(dist.entropy())
        value = critic(state_t)

        state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward 
        done = terminated or truncated
        rewards.append(reward)
        values.append(value)

    mean_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(mean_rewards)} --- ")

    if np.mean(mean_rewards) >= 475:
        print("CartPole Solved!")
        break

    returns = compute_returns(rewards)

    log_probs = torch.stack(log_probs)
    states_t = torch.stack(states)
    values = torch.stack(values)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    entropies = torch.stack(entropies)

    loss_actor = -(log_probs * (returns - values.detach())).mean() - COEF_ENTROPY * entropies.mean()
    optimizer_actor.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), ACTOR_GRAD_NORM)
    optimizer_actor.step()

    loss_critic = (returns - values).pow(2).mean()
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

actor.eval()
critic.eval()

for episode in range(100):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f" --- Episode: {episode}, Total Reward: {total_reward} --- ")

env.close()