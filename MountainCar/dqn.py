import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, size=100_000, batch_size=64, eps=3e-6, alpha=0.5, beta=0.9):
        self.alpha = alpha 
        self.beta = beta
        self.eps = eps
        self.batch_size = batch_size
        self.size = size

        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, next_state, reward, done, td=None):
        if td is None:
            td = 1.0
        self.buffer.append((state, action, next_state, reward, done))

        prio = abs(td + self.eps) ** self.alpha
        self.priorities.append(prio) 

    def sample(self):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False, p=probs)
        batch = [self.buffer[i] for i in idx]
        state, action, next_state, reward, done = zip(*batch)
        
        weights = (self.batch_size * probs[idx]) ** -self.beta
        weights /= weights.max()
        return state, action, next_state, reward, done, idx, weights
    
    def update(self, tds, idx):
        for id, td in zip(idx, tds):
            self.priorities[id] = abs(td + self.eps) ** self.alpha

class DeepQNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.advantage = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x) -> torch.Tensor:
        y = self.shared(x)
        a = self.advantage(y)
        v = self.value(y)
        r = v + (a - a.mean(dim=1, keepdim=True))
        return r
    
class DQNAgent:
    def __init__(self, gamma=0.99, lr=1e-3, hidden=128, warmup=1000, tau=0.05, batch_size=64, buffer_size=100_000, alpha=0.5, beta=0.9, eps_buffer=3e-6, target_update=1000, eps=1.0, eps_decay=0.995, eps_min=0.05, eps_clip=0.2, max_grad_norm=1.0):
        env = gym.make("MountainCar-v0")
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n 
        env.close()

        self.gamma = gamma
        self.lr = lr
        self.hidden = hidden
        self.target_update = target_update
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.warmup = warmup
        self.tau = tau

        self.policy = DeepQNetwork(n_states, n_actions, hidden)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.target_net = DeepQNetwork(n_states, n_actions, hidden)
        self.target_net.load_state_dict(self.policy.state_dict())

        self.buffer = ReplayBuffer(buffer_size, batch_size, eps_buffer, alpha, beta)

    def polyak_update(self):
        for param, tgt_param in zip(self.policy.parameters(), self.target_net.parameters()):
            tgt_param.data.copy_(param * self.tau + tgt_param * (1.0 - self.tau))

    def learn(self, env: gym.Env, timesteps: int = 300_000):
        self.policy.train()
        self.target_net.eval()

        mean_reward = deque(maxlen=100)
        total_reward = 0.0
        state, _ = env.reset()

        for step in range(1,1+timesteps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if np.random.rand() < self.eps:
                action = env.action_space.sample()
            else:
                action = self.policy.forward(state_t).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            self.buffer.append(state, action, next_state, reward, done)

            state = next_state

            self.train()
            self.polyak_update()
            self.eps = max(self.eps_min, self.eps - (1.0 - self.eps_min) / 50_000)
                
            if step % (50 * 200) == 0:
                print(f" --- step: {step}, total reward: {total_reward}, mean reward: {np.mean(mean_reward)} --- ")

            if done:
                mean_reward.append(total_reward)
                total_reward = 0.0
                state, _ = env.reset()

        env.close()

    def train(self):
        if len(self.buffer) < self.warmup:
            return
        
        state, action, next_state, reward, done, idx, weights = self.buffer.sample()

        states_t = torch.from_numpy(np.array(state)).float()
        actions_t = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        next_states_t = torch.from_numpy(np.array(next_state)).float()
        rewards_t = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy.forward(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions_t = self.policy.forward(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net.forward(next_states_t).gather(1, next_actions_t)

        td_error = rewards_t + self.gamma * next_q_values * (1.0 - dones_t) - q_values

        loss = (td_error.pow(2) * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.update(td_error.detach().numpy().flatten(), idx)

    def evaluate(self, env: gym.Env, episodes=10):
        self.policy.eval()

        for episode in range(episodes):
            total_reward = 0.0
            done = False
            state, _ = env.reset()

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.policy.forward(state_t).argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            print(f" --- Episode: {episode}, Total reward: {total_reward} --- ")

        env.close()

if __name__ == '__main__':
    model = DQNAgent()
    env_train = gym.make("MountainCar-v0")
    model.learn(env_train, 100000)
    env_eval = gym.make("MountainCar-v0", render_mode="human")
    model.evaluate(env_eval)