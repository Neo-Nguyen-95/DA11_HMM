import numpy as np

def alpha(t, pi, A, B, O_index):
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
        return np.dot(A.T, alpha(t-1, pi, A, B, O_index)) * b
    
def P_O_given_lambda(pi, A, B, O_index):
    return alpha(len(O_index)-1, pi, A, B, O_index).sum()