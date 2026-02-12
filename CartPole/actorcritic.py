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
TRAINING = 3000
EVAL = 100
LR_ACTOR = 2e-3
LR_CRITIC = 3e-3
COEF_ENTROPY = 0.01
GRAD_NORM_ACTOR = 3.0

class Actor(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, f_out),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    
class Critic(nn.Module):
    def __init__(self, f_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)
    
actor = Actor(n_states, n_actions)
optim_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic = Critic(n_states)
optim_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

actor.train()
critic.train()
mean_rewards = deque(maxlen=100)

for episode in range(TRAINING):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    states = []
    next_states = []
    log_probs = []
    rewards = []
    entropies = []
    dones = []

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32)
        probs = actor(state_t.unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        entropies.append(dist.entropy())

        log_probs.append(dist.log_prob(action))
        states.append(state_t)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        next_states.append(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
        rewards.append(reward)
        dones.append(done)

        state = next_state

    mean_rewards.append(total_reward)
        
    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(mean_rewards):.3f} --- ")

    log_probs = torch.stack(log_probs)
    states_t = torch.stack(states)
    next_states_t = torch.stack(next_states)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32)

    values = critic(states_t)
    next_values = critic(next_states_t).detach()

    td_error = rewards + GAMMA * next_values * (1 - dones) - values
    loss_critic = (td_error).pow(2).mean()

    advantage = td_error.detach()

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    entropy = torch.stack(entropies).mean()

    loss_entropy = COEF_ENTROPY * entropy

    loss_actor = -(log_probs * advantage.squeeze()).mean() - loss_entropy

    optim_actor.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_NORM_ACTOR)
    optim_actor.step()

    optim_critic.zero_grad()
    loss_critic.backward()
    optim_critic.step()

actor.eval()

for episode in range(EVAL):
    state, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    if episode % 10 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward} ---")

env.close()