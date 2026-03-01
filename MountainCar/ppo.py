import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class RolloutBuffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def append(self, state, action, reward, done, logprob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, n_states, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)
    
class PPOAgent:
    def __init__(self, gamma=0.99, lambda_gae=0.95, lr_critic=3e-4, lr_actor=3e-4, batch_size=64, epochs=10, rollout_steps=4096, hidden=128, entropy_coef=0.02, value_loss_coef=0.5, eps_clip=0.2, grad_norm=1.0):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.batch_size = batch_size
        self.epochs = epochs
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.eps_clip = eps_clip
        self.grad_norm = grad_norm

        env = gym.make("MountainCar-v0")
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n 
        env.close()

        self.actor = ActorNetwork(n_states, n_actions, hidden)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = CriticNetwork(n_states, hidden)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def compute_gae(self, next_value):
        advantages = []
        gae = 0.0
        values = torch.cat(self.buffer.values + [next_value])
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)

        for t in reversed(range(len(rewards))):
            td = rewards[t] + self.gamma * values[t+1] * (1.0 - dones[t]) - values[t]
            gae = self.gamma * self.lambda_gae * gae * (1.0 - dones[t]) + td
            advantages.insert(0, gae)

        advantages = torch.stack(advantages).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def collect_rollouts(self, env: gym.Env, max_steps=100_000):
        state, _ = env.reset()
        mean_rewards = deque(maxlen=100)
        total_reward = 0.0
        
        for step in range(1, 1+max_steps):
            state = normalize_state(state)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(state_t)
            value = self.critic(state_t).view(-1)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            self.buffer.append(state, action.item(), reward, done, logprob, value)

            state = next_state

            if done:
                state, _ = env.reset()
                mean_rewards.append(total_reward)

                if step % (200 * 50) == 0:
                    print(f" step: {step}, total reward: {total_reward}, mean reward: {np.mean(mean_rewards)} --- ")
                
                total_reward = 0.0

            if step % self.rollout_steps == 0:

                if done:
                    next_value = torch.zeros(1)
                else:
                    with torch.no_grad():
                        state = normalize_state(state)
                        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        next_value = self.critic(state_t).view(-1)

                advantages = self.compute_gae(next_value)
                

                states_t = torch.from_numpy(np.array(self.buffer.states)).float()
                actions_t = torch.from_numpy(np.array(self.buffer.actions)).to(torch.int64)
                logprobs_t = torch.stack(self.buffer.logprobs).float().squeeze(-1)
                values_t = torch.cat(self.buffer.values).float().squeeze(-1)

                returns = advantages + values_t

                for _ in range(self.epochs):
                    size = states_t.size(0)
                    idx = torch.randperm(size)

                    for start in range(0, size, self.batch_size):
                        end = start+self.batch_size
                        mb_idx = idx[start:end]

                        mb_states = states_t[mb_idx]
                        mb_actions = actions_t[mb_idx]
                        mb_logprobs = logprobs_t[mb_idx].detach()
                        mb_values = values_t[mb_idx].detach()
                        mb_advantages = advantages[mb_idx].detach()
                        mb_returns = returns[mb_idx].detach()


                        logits = self.actor(mb_states)
                        dist = torch.distributions.Categorical(logits=logits)
                        new_logprobs = dist.log_prob(mb_actions)
                        entropies = dist.entropy()
                        new_values = self.critic(mb_states)

                        ratio = torch.exp(new_logprobs - mb_logprobs)
                        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                        surr1 = ratio * mb_advantages
                        actor_loss = -torch.min(surr1, surr2).mean() - entropies.mean() * self.entropy_coef

                        self.optim_actor.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                        self.optim_actor.step()

                        new_values = new_values.view(-1)
                        value_clipped = torch.clamp(new_values - mb_values, -self.eps_clip, self.eps_clip) + mb_values
                        clipped_loss = (mb_returns - value_clipped).pow(2)
                        unclipped_loss = (mb_returns - new_values).pow(2)
                        critic_loss = torch.max(clipped_loss, unclipped_loss).mean() * self.value_loss_coef

                        self.optim_critic.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                        self.optim_critic.step()

                self.buffer.clear()

    def learn(self, max_steps=100_000):
        self.actor.train()
        self.critic.train()
        env = gym.make("MountainCar-v0")
        
        self.collect_rollouts(env, max_steps)

    def evaluate(self, episodes=10):
        env = gym.make("MountainCar-v0", render_mode="human")

        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                state = normalize_state(state)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.actor(state_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                state, reward, terminated, truncated, _ = env.step(action.item())

                total_reward += reward
                done = terminated or truncated

            print(f" --- episode: {episode}, total reward: {total_reward} --- ")

        env.close()

def normalize_state(state):
    x, v = state
    xnorm = (x + 1.2) / (0.6 + 1.2)
    vnorm = (v + 0.07) / (0.07 + 0.07)
    return np.array([xnorm, vnorm], dtype=np.float32)

if __name__ == '__main__':
    model = PPOAgent()
    model.learn(400_000)
    model.evaluate()