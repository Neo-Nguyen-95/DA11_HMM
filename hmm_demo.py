#%% LIBRARY & INPUT
import numpy as np
import pandas as pd
from module_hmm_general import HMM
from module_baum_welch import BaumWelchAlgo

#%% HMM module
# Innitial probability
# π =  [ U ] 
#      [ L ]
pi = [[0.8],
      [0.2]]

# Transition probability
#            U     L 
# A =   U | 0.5 | 0.5 |
#       L | 0   |  1  |
A = [[0.5, 0.5], 
     [0, 1]]

# Emission probability
#           incorrect   correct  (sequence for mapping with index dict M)
# B =   U |    0.9    |   0.1   |
#       L |    0.2    |   0.8   |
B = [[0.9, 0.1],
     [0.2, 0.8]]
M = {'incorrect': 0,
     'correct': 1}

# Observation 
O = pd.Series(
    ['incorrect', 'correct', 'incorrect', 'incorrect', 'incorrect',
     'incorrect','correct', 'incorrect', 'incorrect', 'correct',
     'incorrect','correct', 'incorrect', 'correct', 'correct',
     'correct'])

O_index = np.array(O.map(M))

#%% FORWARD - BACKWARD ALGO
hmm = HMM(pi, A, B, O_index)
hmm.alpha(2)
hmm.beta(6)

# alpha path
hmm.P_O_from_alpha()

# beta path
hmm.P_O_from_beta()

# Combine 2 path
a_side = hmm.alpha(5) * hmm.A
b_side = hmm.beta(6) * hmm.B[:, O_index[6]].reshape(-1, 1)
P_O_from_alpha_beta = np.dot(a_side, b_side).sum()

#%% VITERBI ALGORITHM
hmm.viterbi()

#%% ZETA
hmm.zeta(0)
hmm.gamma(0)


