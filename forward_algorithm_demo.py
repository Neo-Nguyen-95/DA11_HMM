#%% LIBRARY & INPUT
import numpy as np
import pandas as pd

# Innitial probability
# π =  [ U ] 
#      [ L ]
pi = np.array([[0.8],
               [0.2]])

# Transition probability
#            U     L 
# A =   U |    |     |
#       L |    |     |
A = np.array([[0.5, 0.5],
              [0, 1]])

# Emission probability
#           incorrect   correct  (sequence for mapping with index dict M)
# B =   U |           |         |
#       L |           |         |
B = np.array([[0.9, 0.1],
              [0.2, 0.8]])
M = {'incorrect': 0,
     'correct': 1}

# Observation 
O = pd.Series(['correct', 'incorrect', 'incorrect', 'correct', 'correct',
     'incorrect','correct', 'correct', 'incorrect', 'correct',
     'incorrect','correct', 'correct', 'correct', 'correct',
     'correct'])


O_index = np.array(O.map(M))

#%% SUPPORT FUNCTION
def alpha(t, A, B, O_index):
    '''
    Parameter:
        t: alpha at time t
        A: transition probability matrix
        B: emission probability matrix
        O_index: index sequence of observation
        
    '''
    
    b = B[:, O_index[t]].reshape(-1, 1)
  
    if t == 0:        
        return pi * b
    else:
        return np.dot(A.T, alpha(t-1, A, B, O_index)) * b
    
def P_O_given_lambda(A, B, O_index):
    return alpha(len(O_index)-1, A, B, O_index).sum()

#%% CALCULATION

alpha(0, A, B, O_index)
alpha(1, A, B, O_index)

P_O_given_lambda(A, B, O_index)
