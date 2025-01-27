import numpy as np

class BaumWelchAlgo:
    def __init__(self, O_index, N):
        self.O_index = np.array(O_index)
        self.psi = np.arange(N).reshape(-1, 1)
        self.N = N
        self.num_obs = len(set(O_index))
        
        
        pi_rand = np.random.randint(1, 10, size=(self.N, 1))
        # self.pi = pi_rand / np.sum(pi_rand).reshape(-1, 1)
        self.pi = np.array([[1], [0]])
        
        A_rand = np.random.randint(1, 10, size=(self.N, self.N))
        self.A = A_rand / np.sum(A_rand, axis=1).reshape(-1, 1)
        
      
        
        B_rand = np.random.randint(1, 10, size=(self.N, self.num_obs))
        self.B = B_rand / np.sum(B_rand, axis=1).reshape(-1, 1)
        

        
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
        
    def training(self, epochs = 150):
        for _ in range(epochs):          
            
            self.pi = self.gamma(0)
            # print(self.pi)
            
            zeta_sum = 0
            gamma_sum = 0
            for t in range(len(self.O_index)-1):
                zeta_sum += self.zeta(t)
                gamma_sum += self.gamma(t)
                
            self.A = zeta_sum /gamma_sum
            
            # print(self.A)
            
            gamma_sum_2 = gamma_sum + self.gamma(len(self.O_index)-1)
            
            for obs in set(self.O_index):
                
                b_k = np.zeros([self.N, 1])  # temperatory value 
                
                # print('obs index=', np.where(self.O_index==obs)[0])
                for t in np.where(self.O_index==obs)[0]:
                    
                    b_k += self.gamma(t) / gamma_sum_2
                
                # print('B=', self.B)
                # print('b_k=', b_k.reshape(1, -1))
                
                self.B[:, obs] = b_k.reshape(1, -1)
                
                # print('B updated=', self.B)
        
            
            
#%% TEST 
import pandas as pd

df = pd.read_csv('generated_data.csv')
         
bwa = BaumWelchAlgo(O_index=df['obs_index'], N=2)

bwa.training()

bwa.show_value() 


#%%
# df = pd.read_csv('data_python.csv')
# bwa = BaumWelchAlgo(O_index=df['Visible'], N=2)
# bwa.show_value() 

# bwa.training()
# bwa.show_value() 
            
            
            
            
            
            