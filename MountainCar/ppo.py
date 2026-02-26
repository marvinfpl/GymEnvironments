import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def append(self, state, action, reward, done, logp, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logp)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(nn.Linear(hidden, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        y = self.shared(x)
        probs = self.actor(y)
        value = self.critic(y)
        return probs, value
    
    def select_action(self, state): # how to get a general method for checking dimensions?
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, value = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, value, logp

    def evaluate_action(self, state, action):
        state_t = torch.clone(state)
        probs, value = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return value, logp, entropy
    
class PPOAgent():
    def __init__(self, env_name, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, hidden=128, epochs=10, eps_clip=0.2, entropy_coef=0.02, value_loss_coef=0.5, batch_size=64, max_grad_norm=1.0):
        env = gym.make(env_name)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()

        self.policy = ActorCritic(n_states, n_actions, hidden)

        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam([{'params': self.policy.shared.parameters()},
                                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                                {'params': self.policy.critic.parameters(), 'lr': lr_critic}])
        
        self.buffer = RolloutBuffer()

    def compute_returns(self):
        returns = []
        G = 0.0
        for t in reversed(range(len(self.buffer.rewards))):
            if self.buffer.dones[t]:
                G = 0
            G = self.buffer.rewards[t] + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns
    
    def update_policy(self):
        returns = self.compute_returns()

        states_t = torch.from_numpy(np.array(self.buffer.states)).float()
        actions_t = torch.from_numpy(np.array(self.buffer.actions)).to(torch.int64)
        old_logprobs_t = torch.stack(self.buffer.logprobs).float().squeeze()
        values_t = torch.cat(self.buffer.values).float()

        advantage = returns - values_t.squeeze()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.epochs):
            idx = torch.randperm(states_t.size(0))

            for start in range(0, states_t.size(0), self.batch_size):
                end = start+self.batch_size
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_logprobs_t[mb_idx]
                mb_advantage = advantage[mb_idx]
                mb_returns = returns[mb_idx]
                
                value, new_logp, entropy = self.policy.evaluate_action(mb_states, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = (mb_returns.detach() - value.squeeze()).pow(2).mean() * self.value_loss_coef

                loss = actor_loss + critic_loss - entropy.mean() * self.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

if __name__ == '__main__':
    TRAINING = 1000
    EVAL = 10
    rewards = deque(maxlen=100)

    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    
    model = PPOAgent(env_name)
    ac = model.policy

    for episode in range(TRAINING): 
        total_reward = 0.0
        state, _ = env.reset()
        done = False

        while not done:
            action, value, logp = ac.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            model.buffer.append(state, action.item(), reward, done, logp.detach(), value.detach())

            state = next_state

        model.update_policy()

        rewards.append(total_reward)

        if episode % 50 == 0:
            print(f" --- episode: {episode}, total reward: {total_reward}, mean reward: {np.mean(rewards):.3f} --- ")
        
    env.close()

    env = gym.make(env_name, render_mode='human')
    for episode in range(EVAL):
        total_reward = 0.0
        done = False
        state, _ = env.reset()

        while not done:
            action, _, _ = ac.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
        
        print(f" --- Episode: {episode}, total reward: {total_reward} --- ")
    env.close()