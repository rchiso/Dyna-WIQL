import numpy as np
import matplotlib.pyplot as plt
from environment import circulant, RMABEnv
from simulator import run_WIQL, run_Dyna, avg_cum_sum


all_population  = 5
number_states   = 4
number_actions  = 2
budget          = 1
n_instances     = 1
n_episodes      = 1000
seeds           = 10
active, passive, reward_dist = circulant()

total = np.zeros(n_episodes)
#total_idx = np.zeros((all_population,4))
total_dyna1 = np.zeros(n_episodes)
#total_idx_dyna1 = np.zeros((all_population,4))
total_dyna2 = np.zeros(n_episodes)
#total_idx_dyna2 = np.zeros((all_population,4))
total_dyna3 = np.zeros(n_episodes)
#total_idx_dyna3 = np.zeros((all_population,4))


for i in range(seeds):

  
  
  simulator = RMABEnv(all_population=all_population, active_transition=active, passive_transition=passive, budget=budget,number_states=number_states,reward_dist = reward_dist)
  total_reward, indices = run_WIQL(env=simulator,epochs=n_instances,n_episodes=n_episodes)
  total += total_reward
  #total_idx += indices

  total_reward, indices = run_Dyna(env=simulator,epochs=n_instances,n_episodes=n_episodes, offset=0.1, freq=5, duration=10)
  total_dyna1 += total_reward
  #total_idx_dyna1 += indices

  total_reward, indices = run_Dyna(env=simulator,epochs=n_instances,n_episodes=n_episodes, offset=0.3, freq=5, duration=10)
  total_dyna2 += total_reward
  #total_idx_dyna2 += indices

  total_reward, indices = run_Dyna(env=simulator,epochs=n_instances,n_episodes=n_episodes, offset=0.5, freq=5, duration=10)
  total_dyna3 += total_reward
  #total_idx_dyna3 += indices

  print(f'Trial {i+1} done!')


total /= seeds
#total_idx /=seeds
total_dyna1 /= seeds
#total_idx_dyna1 /= seeds
total_dyna2 /= seeds
#total_idx_dyna2 /= seeds
total_dyna3 /= seeds
#total_idx_dyna3 /= seeds


plt.plot(avg_cum_sum(total),label='WIQL')
plt.plot(avg_cum_sum(total_dyna1),label='Dyna-WIQL,offset=0.1')
plt.plot(avg_cum_sum(total_dyna2),label='Dyna-WIQL,offset=0.3')
plt.plot(avg_cum_sum(total_dyna3),label='Dyna-WIQL,offset=0.5')
plt.ylabel('Average cumulative reward')
plt.xlabel('Timesteps')
plt.legend()
plt.title(f'{all_population} arms, {budget} budget')
plt.show()


