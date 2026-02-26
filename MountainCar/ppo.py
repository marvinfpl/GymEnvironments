import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import wandb

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
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
            #nn.Softmax(dim=-1),
        )

        # initialize q_values

        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # initialize state values

    def forward(self, x):
        probs = self.actor(x)
        state_values = self.critic(x)
        return probs, state_values
    
    def select_action(self, state): # how to get a general method for checking dimensions?
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, value.view(-1), logp

    def evaluate_action(self, state, action):
        state_t = torch.clone(state)
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return value.view(-1), logp, entropy
    
    def deterministic_policy(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state_t)
        action = logits.argmax().item()
        return action, value.view(-1)

    
class PPOAgent():
    def __init__(self, env_name, gamma=0.99, lambda_gae=0.98, lr_actor=3e-4, lr_critic=3e-4, hidden=128, epochs=10, rollout_steps=2048, eps_clip=0.2, entropy_coef=0.02, value_loss_coef=0.5, batch_size=64, max_grad_norm=1.0):
        self.env_name = env_name
        env = gym.make(env_name)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()

        self.policy = ActorCritic(n_states, n_actions, hidden)

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epochs = epochs
        self.rollout_steps = rollout_steps
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.optim_actor = optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        
        self.buffer = RolloutBuffer()

        # wandb logger?

    def train(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def eval(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    def compute_gae(self, next_value):
        gae = 0.0
        advantages = []
        values = torch.cat(self.buffer.values + [next_value])
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)
        for t in reversed(range(len(rewards))):
            td = rewards[t] + self.gamma * values[t+1] * (1.0 - dones[t]) - values[t]
            gae = self.gamma * self.lambda_gae * gae * (1.0 - dones[t]) + td
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    
    def update_policy(self, next_value):
        advantages = self.compute_gae(next_value)

        states_t = torch.from_numpy(np.array(self.buffer.states)).float()
        actions_t = torch.from_numpy(np.array(self.buffer.actions)).to(torch.int64)
        old_logprobs_t = torch.stack(self.buffer.logprobs).float().squeeze(-1)
        values_t = torch.cat(self.buffer.values).float().squeeze(-1)

        returns = advantages + values_t

        for _ in range(self.epochs):
            idx = torch.randperm(states_t.size(0))

            for start in range(0, states_t.size(0), self.batch_size):
                end = start+self.batch_size
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_logprobs_t[mb_idx].detach()
                mb_old_values = values_t[mb_idx].detach()
                mb_advantage = advantages[mb_idx].detach()
                mb_returns = returns[mb_idx].detach()
                
                new_values, new_logp, entropy = self.policy.evaluate_action(mb_states, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantage
                actor_loss = -torch.min(surr1, surr2).mean() - entropy.mean() * self.entropy_coef

                self.optim_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.optim_actor.step()

                new_values = new_values.view(-1)
                clipped_values = torch.clamp(new_values - mb_old_values, -self.eps_clip, self.eps_clip) + mb_old_values
                clipped_loss = (mb_returns - clipped_values).pow(2)
                unclipped_loss = (mb_returns - new_values).pow(2)
                critic_loss = torch.max(unclipped_loss, clipped_loss).mean() * self.value_loss_coef

                self.optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.optim_critic.step()

        self.buffer.clear()

    def learn(self, env: gym.Env, total_timesteps=100_000):
        self.train()
        state, _ = env.reset()
        total_reward = 0.0
        mean_reward = deque(maxlen=100)

        for step in range(1, total_timesteps):
            state = normalize_state(state)
            action, value, logp = self.policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            self.buffer.append(state, action.item(), reward, done, logp, value)

            state = next_state

            if done:
                state, _ = env.reset()
                mean_reward.append(total_reward)
                if step % (200 * 50) == 0:
                    print(f" --- step: {step}, total reward: {total_reward}, mean reward: {np.mean(mean_reward)} --- ")
                total_reward = 0.0

            if step %  self.rollout_steps == 0:
                if done:
                    next_value = torch.zeros(1)
                else:
                    with torch.no_grad():
                        state = normalize_state(state)
                        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        _, next_value = self.policy(state_t)
                        next_value = next_value.view(-1)
                self.update_policy(next_value)

        env.close()

    def test(self, env: gym.Env, episodes=10):
        self.eval()
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False

            # stochastic policy
           
            while not done:
                state = normalize_state(state)
                action, _, _ = self.policy.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action.item())
                total_reward += reward
                done = terminated or truncated
            
            # deterministic policy
            """
            while not done:
                state = normalize_state(state)
                action, _ = self.policy.deterministic_policy(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            """

            print(f" --- Episode: {episode}, Total reward: {total_reward} --- ")

        env.close()

def normalize_state(state):
    x, v = state
    xnorm = (x + 1.2) / (0.6 + 1.2)
    vnorm = (v + 0.07) / (0.07 + 0.07)
    return np.array([xnorm, vnorm], dtype=np.float32)

if __name__ == '__main__':
    STEPS = 400_000
    EVAL = 10
    rewards = deque(maxlen=100)

    env_name = "MountainCar-v0"
    env_train = gym.make(env_name)
    env_eval = gym.make(env_name, render_mode='human')
    
    model = PPOAgent(env_name, rollout_steps=4096, entropy_coef=0.005)
    model.learn(env_train, STEPS)
    model.test(env_eval)

