import torch
import torch.nn as nn

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

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = self.shared(x)
        a = self.advantage(y)
        v = self.value(y)
        return v + (a - a.mean(dim=1, keepdim=True))
    
class RewardNetwork(nn.Module):
    def __init__(self, n_states, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)