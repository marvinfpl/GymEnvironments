import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

GAMMA = 0.99
TRAINING = 2500
EVAL = 100
LR = 1e-3
UPDATE = 75
BATCH_SIZE = 64
EPS = 0.995
EPS_DECAY = 0.95
MIN_EPS = 0.005

buffer = deque(maxlen=10000)

env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n 

class DeepQNetwork(nn.Module):
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
        return self.net(x)
    
dqn = DeepQNetwork(n_states, n_actions)
optimizer = optim.Adam(dqn.parameters(), lr=LR)
target_network = DeepQNetwork(n_states, n_actions)
target_network.load_state_dict(dqn.state_dict())

def sample():
    if len(buffer) < BATCH_SIZE:
        return
    
    idx = np.random.choice(len(buffer), BATCH_SIZE)
    batch = [buffer[i] for i in idx]

    state, action, reward, next_state, done = zip(*batch)
    return state, action, reward, next_state, done

def train():
    if len(buffer) < BATCH_SIZE:
        return
    
    state, action, reward, next_state, done = sample()

    states = torch.tensor(state, dtype=torch.float32)
    actions = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
    next_states = torch.tensor(next_state, dtype=torch.float32)
    rewards = torch.tensor(reward, dtype=torch.float32)
    dones = torch.tensor(done, dtype=torch.float32)

    q_values = dqn(states)
    q_value = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_online = dqn(next_states)
        next_actions = next_q_online.argmax(1)

        next_q_target = target_network(next_states)
        next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target = rewards + GAMMA * next_q_value * (1.0 - dones)

    loss = nn.MSELoss()(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


rewards = deque(maxlen=100)
for episode in range(TRAINING):
    state, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        if np.random.rand() < EPS:
            action = env.action_space.sample()
        else:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = dqn(state_t).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        buffer.append((state, action, reward, next_state, done))

        state = next_state

        train()

    rewards.append(total_reward)

    if episode % UPDATE == 0:
        target_network.load_state_dict(dqn.state_dict())

    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(rewards)} --- ")

    EPS = max(EPS*EPS_DECAY, MIN_EPS)

    if np.mean(rewards) >= 450:
        print("CartPole solved!")

dqn.eval()

for episode in range(100):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = dqn(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f" --- Episode: {episode}, Total Reward: {total_reward} --- ")

env.close()