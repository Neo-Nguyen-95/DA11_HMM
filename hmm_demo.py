#%% LIBRARY & INPUT
import numpy as np
import pandas as pd
from module_hmm_general import HMM
from module_baum_welch import BaumWelchAlgo

#%% PART I: HMM PARAMETERS ARE GIVEN
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

# FORWARD - BACKWARD ALGO
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

np.sum(hmm.alpha(3)*hmm.beta(3))

# VITERBI ALGORITHM
hmm.viterbi()

# ZETA
hmm.zeta(0)
hmm.gamma(0)

#%% PART II: ESTIMATED PARAMETERS FROM BAUM-WELCH ALGO
#%% EXPERIMENT 1: EDU DATASET
df = pd.read_csv('generated_edu_data.csv')

# Setting 1.1: 
bwa11 = BaumWelchAlgo(O_index=df['obs_index'], 
                    N=2, 
                    pi0=[[0.8], 
                         [0.2]],
                    A0=[[0.4, 0.6],
                        [0.4, 0.6]],
                    B0=[[0.3, 0.7],
                        [0.3, 0.7]],
                    max_epoch=300
                    )
bwa11.training()
bwa11.show_value()
bwa11.plot_log_likelihood()

hmm11 = HMM(bwa11.pi, bwa11.A, bwa11.B, bwa11.O_index)
hmm11.viterbi()
    
# Setting 1.2:
bwa12 = BaumWelchAlgo(O_index=df['obs_index'], 
                    N=2
                    )
bwa12.show_value()

bwa12.training()

bwa12.show_value()
bwa12.plot_log_likelihood()

hmm12 = HMM(bwa11.pi, bwa11.A, bwa11.B, bwa11.O_index)
hmm12.viterbi()


#%% EXPERIMENT 2: WEATHER
df = pd.read_csv('generated_weather_data.csv')
bwa2 = BaumWelchAlgo(O_index=df['obs_index'],
                    N=2,
                    ll_eps=1e-6,
                    max_epoch=500
                    )

bwa2.show_value()

bwa2.training()

bwa2.show_value()
bwa2.plot_log_likelihood()

# check likelihood from estimated data
pi = [[1.00000000e+00],
      [1.57075949e-18]]
A = [[0.60466981, 0.39533019],
     [0.13771585, 0.86228415]]
B = [[2.59281692e-01, 7.40689429e-01, 2.06481865e-05],
     [5.49740419e-01, 2.43507226e-01, 2.06755469e-01]]
df = pd.read_csv('generated_weather_data.csv')
O_index = df['obs_index']
hmm = HMM(pi, A, B, O_index)
hmm.P_O_from_alpha()


# check probability of original data
pi = [[1], 
      [0]]
A = [[0.35, 0.65],
     [0.25, 0.75]]
B = [[0.8, 0.05, 0.15],
     [0.35, 0.45, 0.2]]
df = pd.read_csv('generated_weather_data.csv')
O_index = df['obs_index']
hmm = HMM(pi, A, B, O_index)
hmm.P_O_from_alpha()

# double check with hmm library
# from hmmlearn import hmm
# model = hmm.CategoricalHMM(n_components=2, n_iter=900)
# model.fit(np.array(O_index).reshape(-1, 1))
# model.emissionprob_
# model.transmat_
# model.startprob_
