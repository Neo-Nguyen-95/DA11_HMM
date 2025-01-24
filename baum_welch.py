import numpy as np

class BaumWelchAlgo:
    def __init__(self, O_index, N):
        self.O_index = np.array(O_index)
        self.psi = np.arange(N).reshape(-1, 1)
        self.N = N
        self.num_obs = len(set(O_index))
        
        self.pi = np.ones([self.N, 1]) / self.N
        self.A = np.ones([self.N, self.N]) / self.N
        self.B = np.ones([self.N, self.num_obs]) / self.num_obs
        
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
        return self.alpha(t) * self.beta(t) / np.sum(self.alpha(t) * self.beta(t))
        
    def training(self, epochs = 50):
        for _ in range(epochs):
            self.pi = self.gamma(0)
            # print(self.pi)
            
            zeta_sum = 0
            gamma_sum = 0
            for t in range(len(self.O_index)-1):
                zeta_sum += self.zeta(t)
                gamma_sum += self.gamma(t)
                
            self.A = zeta_sum /gamma_sum
            
            gamma_sum_2 = gamma_sum + self.gamma(len(self.O_index)-1)
            
            for obs in set(self.O_index):
                b_k = np.zeros([self.N, 1])  # temperatory value 
                
                for t in np.where(self.O_index==obs)[0]:
                    b_k += self.gamma(t) / gamma_sum_2
                
                self.B[:, obs] = b_k.reshape(1, -1)
            
            
            
            
            
            
            
            
            
            