import numpy as np
import pandas as pd

#%% CLASS

class HMMGenerator:
    def __init__(self, pi, A, B, hidden_states, unique_obs_set, obs_len):
        '''
        Parameter:
            
            π: initial probability of hidden states
            π =  state_1 [ 0.7 ] 
                 state_2 [ 0.6 ]
                 
            A: transition probability matrix
                            state_1    state_2
            A = state_1 [     0.3       0.7      ]
                state_2 [     0.4       0.6      ]
            
            B: emission probability matrix
                            obs_1       obs_2
            B = state_1 [     0.2         0.8   ]
                state_2 [     0.1         0.9   ]
            
            hidden_state: set of hidden states
            hidden_state = [state_1, state_2, ..., state_N]
            
            obs_len: length of generated observations
            obs_len = T
        
        '''
        
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
        self.hidden_states_set = hidden_states_set
        self.unique_obs_set = unique_obs_set
        self.obs_len = obs_len
        
        self.state_index_set = np.arange(len(hidden_states))
        self.obs_index_set = np.arange(len(unique_obs_set))
        
        self.state_mapping = {}
        for key, value in enumerate(hidden_states_set):
             self.state_mapping[key] = value
             
        self.obs_mapping = {}
        for key, value in enumerate(unique_obs_set):
             self.obs_mapping[key] = value
    
    def generate_sequences(self):
        
        
        # Initial state with initial prob
        state_index = np.random.choice(self.state_index_set, p=self.pi.flatten())
        state_index_sequence = [state_index]
        
        # Initial observation from initial state
        obs_index = np.random.choice(self.obs_index_set, p=self.B[state_index])
        obs_index_sequence = [obs_index]
        
        for _ in range(self.obs_len - 1):
            # State at t+1
            state_index = np.random.choice(self.state_index_set, p=self.A[state_index])
            state_index_sequence.append(state_index)
            
            # Observation at t+1
            obs_index = np.random.choice(self.obs_index_set, p=self.B[state_index])
            obs_index_sequence.append(obs_index)
    
        
        state_sequence = pd.Series(state_index_sequence).map(self.state_mapping)
        obs_sequence = pd.Series(obs_index_sequence).map(self.obs_mapping)
        
        return (state_index_sequence, 
                obs_index_sequence,
                state_sequence,
                obs_sequence)
            
        
#%% RUNNING
pi = [[0.8], 
      [0.2]]
A = [[0.85, 0.15],
     [0., 1]]
B = [[0.7, 0.3],
     [0.15, 0.85]]
hidden_states_set =  ['Unlearned', 'Learned']
unique_obs_set = ['Incorrect', 'Correct']
obs_len = 100

hg = HMMGenerator(pi, A, B, hidden_states_set, unique_obs_set, obs_len) 

#%% SHOW VARIABLES
hg.pi
hg.A
hg.B
hg.hidden_states_set
hg.unique_obs_set
hg.obs_len
hg.state_index_set
hg.obs_index_set

hg.state_mapping

#%% GENERATE SEQUENCE
(state_index_sequence, obs_index_sequence, state_sequence, obs_sequence
 ) = hg.generate_sequences()

df = pd.DataFrame(
    {'hidden_state': state_sequence,
     'observation': obs_sequence,
     'state_index': state_index_sequence,
     'obs_index': obs_index_sequence
     })
