
def polyak_averaging(tau, tgt_network, dqn):
    for tgt_param, param in zip(tgt_network.parameters(), dqn.parameters()):
        tgt_param.data.copy_(tau * param.data + (1.0 - tau) * tgt_param.data)

def potential_function(state):
    x, v = state
    xnorm = (x + 1.2) / 1.8
    vnorm = (v + 0.07) / 0.14
    
    position_reward = 10 * xnorm
    velocity_reward = 2 * vnorm ** 2
    """
    threshold_bonus = 0
    if x > -0.4:
        threshold_bonus += 10
    if x > 0.0:
        threshold_bonus += 20
    if x > 0.3:
        threshold_bonus += 30
    """
    return position_reward + velocity_reward #+ threshold_bonus

def reward_shaping(reward, state, next_state, done, gamma):
    if done and next_state[0] >= 0.5:
        return 100.0
    
    shaped = reward + gamma * potential_function(next_state) - potential_function(state)
    
    #if next_state[0] > state[0]:
    #    shaped += 5.0
    
    return shaped
