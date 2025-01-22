#%% LIBRARY & INPUT
import numpy as np
import pandas as pd
from forward_algorithm import alpha, P_O_given_lambda
from viterbi_algorithm import viterbi_algorithm

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
O = pd.Series(['incorrect', 'correct', 'incorrect', 'incorrect', 'incorrect',
     'incorrect','correct', 'incorrect', 'incorrect', 'correct',
     'incorrect','correct', 'incorrect', 'correct', 'correct',
     'correct'])


O_index = np.array(O.map(M))

# Initial state
# Unlearned : 0
# Learned   : 1
psi = [[0],
       [1]]

#%% CALCULATION

alpha(0, pi, A, B, O_index)
alpha(1, pi, A, B, O_index)

P_O_given_lambda(pi, A, B, O_index)

#%% Viterbi algorithm

delta, sequence = viterbi_algorithm(psi, pi, A, B, O_index)
print(delta)
print(sequence)

