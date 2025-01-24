import numpy as np
import pandas as pd
from baum_welch import BaumWelchAlgo

df = pd.read_csv('validate_data.csv')

bwa = BaumWelchAlgo(df['Visible'].head(20), N=2)

bwa.training()

bwa.num_obs

bwa.show_value()

bwa.P_O_from_alpha()
