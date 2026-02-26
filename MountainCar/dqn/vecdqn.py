import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

GAMMA = 0.99
STEPS = 300_000
EVAL = 10
BATCH_SIZE = 64
HIDDEN = 128
LR = 1e-3
N_ENVS = 5
EPS = 1.0
EPS_DECAY = 0.999
EPS_MIN = 0.05
UPDATE = 50
WARMUP = 500

envs = gym.make_vec("MountainCar-v0", num_envs=N_ENVS, vectorization_mode="sync") # is it the same with async?
n_states = envs.single_observation_space.shape[0]
n_actions = envs.single_action_space.n

buffer = deque(maxlen=100000)

def sample():
    idx = np.random.choice(len(buffer), BATCH_SIZE, replace=True)
    batch = [buffer[i] for i in idx]

    states, actions, next_states, rewards, dones = zip(*batch)

    return states, actions, next_states, rewards, dones

class DeepQNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
    
dqn = DeepQNetwork(n_states, n_actions, HIDDEN)
optimizer = optim.Adam(dqn.parameters(), lr=LR)
target_net = DeepQNetwork(n_states, n_actions, HIDDEN)
target_net.load_state_dict(dqn.state_dict())
    
def train():
    if len(buffer) < BATCH_SIZE:
        return
    
    states, actions, next_states, rewards, dones = sample()

    states_t = torch.from_numpy(np.array(states)).float()
    actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    next_states_t = torch.from_numpy(np.array(next_states)).float()
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = dqn(states_t).gather(1, actions_t)

    with torch.no_grad():
        next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)
        td_error = rewards_t + GAMMA * next_q_values * (1.0 - dones_t)

    loss = (td_error - q_values).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
    optimizer.step()

states, _ = envs.reset()
dqn.train()
total_rewards = np.zeros(N_ENVS)

for step in range(STEPS):
    if np.random.rand() < EPS:
        actions = np.array([envs.single_action_space.sample() for _ in range(N_ENVS)], dtype=np.int32)
    else:
        states_t = torch.from_numpy(np.array(states)).float()
        actions = dqn(states_t).argmax(dim=1).numpy().astype(np.int32)

    next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
    total_rewards += rewards

    done_mask = np.array(terminateds) | np.array(truncateds)
    for i in range(N_ENVS):
        buffer.append((states[i], actions[i], next_states[i], rewards[i], done_mask[i]))

    if done_mask.any():
        new_states, _ = envs.reset()
        states[done_mask] = new_states[done_mask]
        total_rewards[done_mask] = np.zeros(N_ENVS)[done_mask]

    if step % UPDATE == 0:
        target_net.load_state_dict(dqn.state_dict())

    states = next_states

    if len(buffer) > WARMUP:
        train()

    EPS = max(EPS * EPS_DECAY, EPS_MIN)

    if step % 500 == 0:
        print(f" --- step: {step}, total reward: {total_rewards}, mean rewards: {total_rewards.mean()} --- ")

dqn.eval()
eval_env = gym.make("MountainCar-v0", render_mode="human")
for episode in range(EVAL):
    state, _ = eval_env.reset()
    total_reward = 0.0
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = dqn(state_t).argmax().item()
        state, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f" --- Episode: {episode}, Total Reward: {total_reward} --- ")

eval_env.close()