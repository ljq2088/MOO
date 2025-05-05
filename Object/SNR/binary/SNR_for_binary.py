import sys
import os
import sys
sys.path.append('/home/ljq/code/MOO')
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from config.config import config
from waveform.binary_waveform import *
from utils.inner_prod import inner_prod
from utils.PSD import PSD_L_lambda
from para_sb import *
for i in np.arange(0,m):
    for j in np.arange(0,n):
        print(L,l)
        PSD_seq=PSD_L_lambda(freq_bin,[L,l])
        SNRtemp = inner_prod(h_f,h_f,PSD_seq,df)
        SNR[i][j]=np.sqrt(SNRtemp)
        
        l=l+dl
    l=l1
    L=L+dL
    


np.savetxt("/home/ljq/code/MOO/results/SNR/binary/SNR_binary3.txt",SNR,fmt="%50.50f",delimiter=" ")