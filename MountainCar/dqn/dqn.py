import torch
import torch.optim as optim
from collections import deque
import wandb 
import numpy as np
from utils import reward_shaping, polyak_averaging
import gymnasium as gym
from network import DeepQNetwork
from buffer import Buffer

envs = gym.make_vec("MountainCar-v0", num_envs=5, vectorization_mode="sync")
n_states = envs.observation_space.shape[0]
n_actions = envs.action_space.n 

state_dist = {}
action_dist = {}

GAMMA = 0.99
TRAINING = 1000
EVAL = 10
UPDATE_TGT = 30
HIDDEN = 128
TAU = 0.05
LR = 2e-3
BUFFER_SIZE = 100_000
ALPHA = 0.6
BETA = 0.9
BATCH_SIZE = 128
CLIP = 2.0
MEAN_REWARD = 100
EPS = 1.0
EPS_DECAY = 0.999
EPS_MIN = 0.05
EPS_BUFFER = 1e-5
WARM_UP = 500

buffer = Buffer(BUFFER_SIZE, BATCH_SIZE, ALPHA, BETA, EPS_BUFFER)

"""
wandb.init(
    project="MountainCar_Analysis",
    config={
        "env": "MountainCar-v0",
        "model": "dueling dqn with PER",
        "gamma": GAMMA,
        "training": TRAINING,
        "eval": EVAL,
        "update_target": UPDATE_TGT,
        "hidden": HIDDEN,
        "tau": TAU,
        "learning_rate": LR,
        "clip_grad_norm": CLIP,
        "mean_reward": MEAN_REWARD,
        "eps": EPS,
        "eps_decay": EPS_DECAY,
        "eps_min": EPS_MIN,
        "warm_up": WARM_UP,
        "buffer": {
            "buffer_size": BUFFER_SIZE,
            "alpha": ALPHA,
            "beta": BETA,
            "batch_size": BATCH_SIZE,
            "eps": EPS_BUFFER,
        }
    }
)
"""
dqn = DeepQNetwork(n_states, n_actions, HIDDEN)
optimizer = optim.Adam(dqn.value.parameters(), lr=LR)
target_net = DeepQNetwork(n_states, n_actions)
target_net.load_state_dict(dqn.state_dict())

def train():
    if len(buffer) < WARM_UP:
        return 0.0
    
    state, action, next_state, reward, done, weights, idx = buffer.sample()

    state_t = torch.from_numpy(np.array(state)).float()
    action_t = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
    next_state_t = torch.from_numpy(np.array(next_state)).float()
    reward_t = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
    done_t = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

    q_values = dqn(state_t).gather(1, action_t)

    with torch.no_grad():
        next_action_t = dqn(next_state_t).argmax(dim=1, keepdim=True)
        next_q_value = target_net(next_state_t).gather(1, next_action_t) 
        td_target = reward_t + GAMMA * next_q_value * (1.0 - done_t)
    td_error = td_target - q_values


    weights_t = torch.tensor(weights, dtype=torch.float32)
    loss = (weights_t * td_error.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), CLIP)
    optimizer.step()

    buffer.update(td_error.detach().numpy().flatten(), idx)

    return loss.item()

mean_old_rewards = deque(maxlen=MEAN_REWARD)
mean_new_rewards = deque(maxlen=MEAN_REWARD)
step = 0

dqn.train()
target_net.eval()

for episode in range(TRAINING):
    state, _ = envs.reset()
    old_total_reward = 0.0
    shaped_total_reward = 0.0
    done = False
    step += 1

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32)

        """
        if state.tolist() not in state_dist: 
            state_dist[state.tolist()] = 1
        else:
            state_dist[state.tolist()] += 1
        """
        if np.random.rand() < EPS:
            action = envs.action_space.sample()
        else:
            action = dqn(state_t).argmax().item()

        """
        if action not in action_dist:
            action_dist[action] = 1
        else:
            action_dist[action] += 1
        """

        next_state, old_reward, terminated, truncated, _ = envs.step(action)
        old_total_reward += old_reward
        done = terminated or truncated

        new_reward = reward_shaping(old_reward, state, next_state, done, GAMMA, True)
        shaped_total_reward += new_reward

        buffer.append((state, action, next_state, new_reward, done))

        state = next_state
        """
        wandb.log(
            data={
                "old_reward": old_reward,
                "new_reward": new_reward,
                "done": done,
                #state_dist
                #action_dist
            }
        )"""
    mean_old_rewards.append(old_total_reward)
    mean_new_rewards.append(shaped_total_reward)

    if episode % UPDATE_TGT == 0:
        polyak_averaging(TAU, target_net, dqn)

    loss = train()

    EPS = max(EPS * EPS_DECAY, EPS_MIN)

    if episode % 50 == 0:
        if loss is None:
            loss = 0.0
        print(f" --- Episode: {episode}, Old Total Reward: {old_total_reward}, Old Mean Reward: {np.mean(mean_old_rewards):.3f}, Shaped Total Reward: {shaped_total_reward:.3f}, Shaped Mean Reward: {np.mean(mean_new_rewards):.3f}, Loss: {loss:.3f}, Eps: {EPS:.3f} --- ")
    """
    wandb.log(
        data={
            "episode": episode,
            "mean_old_reward": np.mean(mean_old_rewards),
            "total_old_reward": old_total_reward,
            "mean_shaped_reward": np.mean(mean_new_rewards),
            "shaped_total_reward": shaped_total_reward,
            "loss": loss,
            "eps": EPS,
        },
        step=step,
    )
    """
            
envs.close()

eval_env = gym.make("MountainCar-v0", render_mode="human")
dqn.eval()

for episode in range(EVAL):
    state, _ = eval_env.reset()
    done = False
    old_total_reward = 0.0
    shaped_total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = dqn(state_t).argmax(dim=1).item()
        next_state, old_reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        new_reward = reward_shaping(old_reward, state, next_state, done, GAMMA, False)
        shaped_total_reward += new_reward
        old_total_reward += old_reward

    print(f" --- Episode: {episode}, Old Total Reward: {old_total_reward}, Shaped Total Reward: {shaped_total_reward} --- ")

    """
    wandb.log(
        data={
            "episode": episode,
            "old_total_reward": old_total_reward,
            "shaped_total_reward": shaped_total_reward,
            }
        )
    """

eval_env.close()