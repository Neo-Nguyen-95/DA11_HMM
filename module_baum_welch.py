import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module_hmm_general import HMM

#%% CLASS
class BaumWelchAlgo:
    def __init__(self, O_index, N, pi0=None, A0=None, B0=None, 
                 ll_eps=1e-6, max_epoch = 200, random_state=42):
        self.O_index = np.array(O_index)
        self.psi = np.arange(N).reshape(-1, 1)
        self.N = N
        self.num_obs = len(set(O_index))
        self.ll_eps = ll_eps
        self.max_epoch = max_epoch
        
        # Store values of π, A, B or
        # Make initial guess if they are not given
        if pi0:
            self.pi = np.array(pi0)
        else:
            np.random.seed(random_state)
            pi_rand = np.random.randint(1, 10, size=(self.N, 1))
            self.pi = pi_rand / np.sum(pi_rand).reshape(-1, 1)
            
        if A0:
            self.A = np.array(A0)
        else:
            np.random.seed(random_state+1)
            A_rand = np.random.randint(1, 10, size=(self.N, self.N))
            self.A = A_rand / np.sum(A_rand, axis=1).reshape(-1, 1)
        
        if B0:
            self.B = np.array(B0)
        else:
            np.random.seed(random_state+2)
            B_rand = np.random.randint(1, 10, size=(self.N, self.num_obs))
            self.B = B_rand / np.sum(B_rand, axis=1).reshape(-1, 1)
        
        # Create a blank list to check probability of HMM parameter each epoch
        self.log_likelihood = []

    def show_value(self):
        # print(self.O_index)
        # print(self.psi)
        
        print(self.pi)
        print(self.A)     
        print(self.B)
        
    def alpha(self, t):
        '''alpha = [α(U)
                    α(L)]
        '''
        
        b = self.B[:, self.O_index[t]].reshape(-1, 1)
    
        if t == 0:        
            return self.pi * b
        else:
            return np.dot(self.A.T, self.alpha(t-1)) * b
        
    def beta(self, t):
        '''beta = [β(U)
                   β(L)]
        '''
        
        T = len(self.O_index)-1
        
        if t == T:
            return np.ones(len(self.pi)).reshape(-1, 1)
        else:
            b_time_beta = self.B[:, self.O_index[t+1]].reshape(-1, 1) * self.beta(t+1)
            return np.dot(self.A, b_time_beta)
        
    def P_O_from_alpha(self):
        return self.alpha(len(self.O_index)-1).sum()
        
    def zeta(self, t):
        '''            U         L
        zeta = U | ζ(U, U) | ζ(U, L) |
               L | ζ(L, U) | ζ(L, L) |
        
        '''
        
        return (self.alpha(t) * self.A 
                * self.B[:, self.O_index[t+1]] * self.beta(t+1).reshape(1, -1)
                / self.P_O_from_alpha()
                )
    
    def gamma(self, t):
        '''
        gamma = γ(U)
                γ(L)
        '''
        return self.alpha(t) * self.beta(t) / self.P_O_from_alpha()
        
    def training(self):
        
        # Just an initial value for log likelihood delta
        ll_delta = 1
        
        # Initial count
        epoch = 0
        
        while ll_delta > self.ll_eps and epoch < self.max_epoch:
            zeta_sum = 0
            gamma_sum = 0
            for t in range(len(self.O_index)-1):
                zeta_sum += self.zeta(t)
                gamma_sum += self.gamma(t)
                
            gamma_sum_2 = gamma_sum + self.gamma(len(self.O_index)-1)  
            B_temp = self.B  # temperature for this variable
            for obs in set(self.O_index):
                
                b_k = np.zeros([self.N, 1])  # temperatory value 
                
                for t in np.where(self.O_index==obs)[0]:
                    
                    b_k += self.gamma(t) / gamma_sum_2
                
                B_temp[:, obs] = b_k.reshape(1, -1)
                
            # update paramater
            self.pi = self.gamma(0)
            self.A = zeta_sum /gamma_sum
            self.B = B_temp
            epoch += 1
                
            # compute current probability
            hmm = HMM(self.pi, self.A, self.B, self.O_index)
            self.log_likelihood.append(np.log10(hmm.P_O_from_alpha()))
            
            # update ll_delta from 3rd cycles
            if epoch > 1:
                ll_delta = abs(self.log_likelihood[-1] - self.log_likelihood[-2])
                
        print('Number of epochs:', epoch)
        print('Current log likelihood delta:', ll_delta)
        
    def plot_log_likelihood(self):
        pd.Series(self.log_likelihood).plot()
        plt.show()
        
          