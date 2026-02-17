from collections import deque
import numpy as np

class Buffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.9, eps=1e-4):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __len__(self):
        return len(self.buffer)

    def append(self, batch, td=None): # batch format: (state, action, next_state, reward, done)
        if td is None:
            td = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(batch)
        self.priorities.append((abs(td) + self.eps) ** self.alpha)

    def sample(self):
        if len(self.buffer) < self.batch_size:
            raise Exception("Batch size is superior to the length of the buffer")
        
        prio = np.array(self.priorities, dtype=np.float32)
        prio = prio + 1e-8
        probs = prio / prio.sum()
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=True, p=probs)
        batch = [self.buffer[i] for i in idx]

        n = len(self.priorities)
        weights = (n * probs[idx]) ** -self.beta
        w = weights / weights.max()

        state, action, next_state, reward, done = zip(*batch)
        return state, action, next_state, reward, done, w, idx
    
    def update(self, tds, idx):
        for td, id in zip(tds, idx):
            self.priorities[id] = (abs(td) + self.eps) ** self.alpha

    

