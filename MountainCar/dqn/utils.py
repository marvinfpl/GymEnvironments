from math import sqrt
import torch
import torch.optim as optim
from network import RewardNetwork

LR = 2e-3

reward_net = RewardNetwork(2, 128)
optimizer = optim.Adam(reward_net.parameters(), lr=LR)

def polyak_averaging(tau, tgt_network, dqn):
    for tgt_param, param in zip(tgt_network.parameters(), dqn.parameters()):
        tgt_param.data.copy_(tau * param.data + (1.0 - tau) * tgt_param.data)

def reward_shaping(reward, state, next_state, done, gamma, training=False):
    if done and next_state[0] >= 0.5:
        return 100.0
    
    if training:
        reward_net.train()
    else:
        reward_net.eval()
    
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    value = reward_net(state_t)
    if done:
        next_value = 0.0
    else:
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_value = reward_net(next_state_t).detach()
    shaped = (reward + gamma * next_value - value).detach()

    td_target = reward + gamma * next_value
    loss = (value - td_target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_net.parameters(), 2.0)
    optimizer.step()

    return shaped.item()
