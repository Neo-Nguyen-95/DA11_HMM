#%% LIBRARY & INPUT
import numpy as np
import pandas as pd
from hmm_module import HMM

# Innitial probability
# π =  [ U ] 
#      [ L ]
pi = np.array([[0.8],
               [0.2]])

# Transition probability
#            U     L 
# A =   U | 0.5 | 0.5 |
#       L | 0   |  1  |
A = np.array([[0.5, 0.5],
              [0, 1]])

# Emission probability
#           incorrect   correct  (sequence for mapping with index dict M)
# B =   U |    0.9    |   0.1   |
#       L |    0.2    |   0.8   |
B = np.array([[0.9, 0.1],
              [0.2, 0.8]])
M = {'incorrect': 0,
     'correct': 1}

# Observation 
O = pd.Series(
    ['incorrect', 'correct', 'incorrect', 'incorrect', 'incorrect',
     'incorrect','correct', 'incorrect', 'incorrect', 'correct',
     'incorrect','correct', 'incorrect', 'correct', 'correct',
     'correct'])

O_index = np.array(O.map(M))

# Initial state
# Unlearned : 0
# Learned   : 1
psi = [[0],
       [1]]

#%% FORWARD - BACKWARD ALGO
hmm = HMM(pi, A, B, O_index, psi)
hmm.alpha(2)
hmm.beta(6)

hmm.P_O_from_alpha()
hmm.P_O_from_beta()

#%% COMBINE 2 PATH
a_side = hmm.alpha(5) * A
b_side = hmm.beta(6) * B[:, O_index[6]].reshape(-1, 1)
P_O_from_alpha_beta = np.dot(a_side, b_side).sum()

#%% VITERBI ALGORITHM
hmm.viterbi()

