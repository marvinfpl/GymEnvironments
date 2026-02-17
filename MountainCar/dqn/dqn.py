import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from buffer import Buffer
from collections import deque

env = gym.make("MountainCar-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n 

GAMMA = 0.99
TRAINING = 1500
EVAL = 100
LR = 3e-3
BATCH_SIZE = 128
EPS = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01
TARGET_UPDATE = 50
AVERAGING_CST = 0.99
BUFFER_SIZE = 10_000
WARM_UP = 400
CLIP = 10

buffer = Buffer(BUFFER_SIZE, BATCH_SIZE)

class DeepQNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)
    
dqn = DeepQNetwork(n_states, n_actions)
optimizer = optim.Adam(dqn.parameters(), lr=LR)
tgt_network = DeepQNetwork(n_states, n_actions)
tgt_network.load_state_dict(dqn.state_dict())

def polyak_averaging():
    for tgt_param, param in zip(tgt_network.parameters(), dqn.parameters()):
        tgt_param.data.copy_(AVERAGING_CST * param.data + (1.0 - AVERAGING_CST) * tgt_param.data)

def potential_function(state):
    x, v = state
    return 10 * x + 1/2 * v ** 2

def reward_shaping(reward, state, next_state):
    return reward + GAMMA * potential_function(next_state) - potential_function(state)

def train():
    if len(buffer) < WARM_UP:
        return
    
    state, action, next_state, reward, done, weight, idx = buffer.sample()

    state_t = torch.tensor(state, dtype=torch.float32)
    next_state_t = torch.tensor(next_state, dtype=torch.float32)
    action_t = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
    reward_t = torch.tensor(reward, dtype=torch.float32)
    done_t = torch.tensor(done, dtype=torch.float32)
    weight_t = torch.tensor(weight, dtype=torch.float32)

    q_values = dqn(state_t).gather(1, action_t)

    with torch.no_grad():
        next_actions = dqn(next_state_t).argmax(dim=1, keepdim=True)
        next_q_values = tgt_network(next_state_t).gather(1, next_actions)
        td_target = reward_t + GAMMA * next_q_values * (1.0 - done_t)

    td_error = td_target - q_values
    td_error = torch.clamp(td_error, -CLIP, CLIP)
    loss = (weight_t * td_error.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    buffer.update(td_error.detach().numpy().flatten(), idx)

rewards = deque(maxlen=100)
dqn.train()
tgt_network.train()

for episode in range(TRAINING):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if np.random.rand() < EPS:
            action = env.action_space.sample()
        else:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = dqn(state_t).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        new_reward = reward_shaping(reward, state, next_state)
        done = terminated or truncated
        total_reward += new_reward
        buffer.append((state, action, next_state, new_reward, done))

        train()

        state = next_state

    rewards.append(total_reward)

    if episode % 50 == 0:
        print(f" --- Episode: {episode}, Total Reward: {total_reward}, Mean Reward: {np.mean(rewards):.3f} --- ")
    
    if np.mean(rewards) > -110:
        print("MountainCar solved!")

    if episode % TARGET_UPDATE == 0:
        polyak_averaging()

    EPS = max(EPS * EPS_DECAY, EPS_MIN)

dqn.eval()

for episode in range(EVAL):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = dqn(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated 

    print(f" --- Episode: {episode}, Total reward: {total_reward} --- ")

env.close()