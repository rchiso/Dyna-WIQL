import numpy as np
import gymnasium as gym

class RMABEnv(gym.Env):


    def __init__(self, all_population, active_transition, passive_transition, budget, number_states, reward_dist):
        '''
        Initialization
        '''

        self.n_arms          = all_population
        self.active          = active_transition
        self.passive         = passive_transition
        #self.cohort_size     = cohort_size
        self.budget          = budget
        self.number_states   = number_states
        #self.episode_len     = episode_len
        self.reward_dist     = reward_dist
        #self.n_instances     = n_instances  # n_epochs: number of separate transitions / instances


        #assert_valid_transition(all_transitions)




    def reset(self):
        self.states = np.random.choice(a=np.arange(self.number_states),size=self.n_arms)
        return self.observe()


    def observe(self):
        return self.states

    def step(self, action):

        next_states = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if action[i] == 1:
              prob = self.active[self.states[i]]
              if max(prob) == 0:
                prob = np.ones(self.number_states)/self.number_states
              next_state = np.random.choice(a=self.number_states, p=prob/sum(prob))
              next_states[i] = next_state
            elif action[i] == 0:
              prob = self.passive[self.states[i]]
              if max(prob) == 0:
                prob = np.ones(self.number_states)/self.number_states
              next_state = np.random.choice(a=self.number_states, p=prob/sum(prob))
              next_states[i] = next_state
        self.states = next_states.astype(int)

        reward = self.get_reward()
        done = False

        # print(f'  action {action}, sum {action.sum()}, reward {reward}')

        return self.observe(), reward, done, {}

    def get_reward(self):
        states = self.observe()
        r = 0
        for i in range(self.n_arms):
          r += self.reward_dist[states[i]]
        return r



def circulant():
  active = np.array([[0.5,0.5,0,0],[0,0.5,0.5,0],[0,0,0.5,0.5],[0.5,0,0,0.5]])
  passive = np.array([[0.5,0,0,0.5],[0.5,0.5,0,0],[0,0.5,0.5,0],[0,0,0.5,0.5]])
  rewards = np.array([-1,0,0,1])
  return active,passive,rewards