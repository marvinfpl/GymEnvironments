import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

GAMMA = 0.99
TRAINING = 1000
EVAL = 100
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
ACTOR_GRAD_NORM = 1.5
CRITIC_GRAD_NORM = 1.5
COEF_ENTROPY = 0.02
BATCH_SIZE = 64

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
            nn.Linear(128, f_out)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1).squeeze()
    
class Critic(nn.Module):
    def __init__(self, f_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)
    
actor = Actor(n_states, n_actions)
optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic = Critic(n_states)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

def compute_n_steps_return(rewards, value):
    G = value
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    return returns

actor.train()
critic.train()
mean_rewards = deque(maxlen=100)

for episode in range(1, 1+TRAINING):
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0

    states = []
    log_probs = []
    values = []
    rewards = []
    entropies = []

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        value = critic(state_t)
        entropy = dist.entropy()

        states.append(state_t)
        values.append(value.unsqueeze(0))
        entropies.append(entropy)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        rewards.append(reward)

        if steps % BATCH_SIZE == 0 or done:
            states_t = torch.cat(states)
            logp = torch.stack(log_probs)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            entropy = torch.stack(entropies).mean()
            values_t = torch.cat(values).unsqueeze(1)
            
            if done:
                next_value = torch.zeros(1, dtype=torch.float32).detach()
            else:
                next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                next_value = critic(next_state_t).detach()

            td_target = compute_n_steps_return(rewards, next_value)
            td_error = td_target - values_t

            loss_critic = td_error.pow(2).mean()
            optimizer_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), CRITIC_GRAD_NORM)
            optimizer_critic.step()
            
            advantage = td_error.detach()
            loss_actor = -(logp * advantage).mean() - COEF_ENTROPY * entropy
            optimizer_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), ACTOR_GRAD_NORM)
            optimizer_actor.step()

            states.clear()
            log_probs.clear()
            entropies.clear()
            rewards.clear()
            values.clear()

        state = next_state

        steps += 1


    mean_rewards.append(total_reward)

    if np.mean(mean_rewards) >= 475.0:
        print('CartPole solved!')

    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(mean_rewards)} --- ")

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

    print(f" --- Episode: {episode}, Total Reward: {total_reward} --- ")

env.close()