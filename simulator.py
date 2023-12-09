import numpy as np
import random as rd
from environment import RMABEnv


def run_WIQL(env,epochs,n_episodes):
  n_arms = env.n_arms
  n_states = env.number_states
  n_actions = 2
  budget = env.budget
  rewards = env.reward_dist
  n_pulls = np.zeros((n_arms,n_states,n_actions))
  indices = np.zeros((n_arms,n_states))
  Q = np.zeros((n_arms,n_states,n_actions))
  env.reset()
  total_reward = np.zeros(n_episodes)
  for ep in range(n_episodes):
    states = env.observe()
    current_indices = np.zeros(n_arms)
    for i in range(n_arms):
      current_indices[i] = indices[i,states[i]]
    epsilon = n_arms/(n_arms+ep)
    if rd.random() < epsilon:
      selection_idx = np.random.choice(a=np.arange(n_arms), size=budget, replace=False)
    else:
      selection_idx = np.argpartition(current_indices, -1*budget)[-1*budget:]

    #selection_idx = np.argmax(indices)
    action = np.zeros(n_arms)
    action[selection_idx] = 1
    action = action.astype(int)
    #print(action)
    #print(states)
    next_states, r, done, _ = env.step(action)
    total_reward[ep] = r
    #print(next_states)
    for i in range(n_arms):
      n_pulls[i][states[i]][action[i]] += 1

    for i in range(n_arms):
      alpha = 1/(1+n_pulls[i,states[i],action[i]])
      Q[i,states[i],action[i]] = (1 - alpha)*Q[i,states[i],action[i]] + alpha*(rewards[states[i]]+ np.max(Q[i,next_states[i]]))
      indices[i,states[i]] = Q[i,states[i],1] - Q[i,states[i],0]
  return total_reward, indices

def run_Dyna(env,epochs,n_episodes, offset, freq, duration):
  n_arms = env.n_arms
  n_states = env.number_states
  n_actions = 2
  budget = env.budget
  rewards = env.reward_dist
  offset = int(offset*n_episodes)
  n_pulls = np.zeros((n_arms,n_states,n_actions))
  n_engaged = np.zeros((n_arms,n_states,n_actions,n_states))
  indices = np.zeros((n_arms,n_states))
  Q = np.zeros((n_arms,n_states,n_actions))
  env.reset()
  total_reward = np.zeros(n_episodes)
  for ep in range(n_episodes):
    states = env.observe()
    current_indices = np.zeros(n_arms)
    for i in range(n_arms):
      current_indices[i] = indices[i,states[i]]
    epsilon = n_arms/(n_arms+ep)
    if rd.random() < epsilon:
      selection_idx = np.random.choice(a=np.arange(n_arms), size=budget, replace=False)
    else:
      selection_idx = np.argpartition(current_indices, -1*budget)[-1*budget:]

    #selection_idx = np.argmax(indices)
    action = np.zeros(n_arms)
    action[selection_idx] = 1
    action = action.astype(int)
    #print(action)
    #print(states)
    next_states, r, done, _ = env.step(action)
    total_reward[ep] = r
    #print(next_states)
    for i in range(n_arms):
      n_pulls[i][states[i]][action[i]] += 1
      for s in range(n_states):
        if next_states[i] == s:
          n_engaged[i,states[i],action[i],next_states[i]] += 1


    for i in range(n_arms):
      alpha = 1/(1+n_pulls[i,states[i],action[i]])
      Q[i,states[i],action[i]] = (1 - alpha)*Q[i,states[i],action[i]] + alpha*(rewards[states[i]]+ np.max(Q[i,next_states[i]]))
      indices[i,states[i]] = Q[i,states[i],1] - Q[i,states[i],0]

    if ep > offset:
      if ep % freq == 0:
        est_passive = np.zeros((n_states,n_states))
        est_active = np.zeros((n_states,n_states))
        for i in range(n_arms):
          for s in range(n_states):
            for s_ in range(n_states):
              if n_pulls[i,s,0] != 0:
                est_passive[s,s_] += n_engaged[i,s,0,s_]/n_pulls[i,s,0]
              if n_pulls[i,s,1] != 0:
                est_active[s,s_] += n_engaged[i,s,1,s_]/n_pulls[i,s,1]
        est_passive /= n_arms
        est_active /= n_arms

        est_env = RMABEnv(all_population=n_arms, active_transition=est_active, passive_transition=est_passive, budget=budget,number_states=n_states,reward_dist = rewards)
        est_env.reset()
        for _ in range(duration):
          est_states = est_env.observe()
          est_current_indices = np.zeros(n_arms)
          for i in range(n_arms):
            est_current_indices[i] = indices[i,est_states[i]]
          epsilon = n_arms/(n_arms+ep)
          #m_epsilon = epsilon
          m_epsilon = 1
          if rd.random() < m_epsilon:
            est_selection_idx = np.random.choice(a=np.arange(n_arms), size=budget, replace=False)
          else:
            est_selection_idx = np.argpartition(est_current_indices, -1*budget)[-1*budget:]

          #selection_idx = np.argmax(indices)
          est_action = np.zeros(n_arms)
          est_action[est_selection_idx] = 1
          est_action = action.astype(int)
          #print(action)
          #print(states)
          est_next_states, r, done, _ = est_env.step(est_action)

          for i in range(n_arms):
            alpha = 1/(1+n_pulls[i,est_states[i],est_action[i]])
            Q[i,est_states[i],est_action[i]] = (1 - alpha)*Q[i,est_states[i],est_action[i]] + alpha*(rewards[est_states[i]]+ np.max(Q[i,est_next_states[i]]))
            indices[i,est_states[i]] = Q[i,est_states[i],1] - Q[i,est_states[i],0]


  return total_reward, indices

def avg_cum_sum(x):
  x = np.cumsum(x)
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = x[i]/(i+1)
  return y
