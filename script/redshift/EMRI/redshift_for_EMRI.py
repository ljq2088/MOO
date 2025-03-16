import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from math import pi as Pi
import matplotlib.pyplot as plt
#%matplotlib inline
from para_re import *
from scipy import interpolate
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from multiprocessing import Pool, cpu_count
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

i=0
j=0

for L in np.arange(L1,L2,dL):
    for l in np.arange(l1,l2,dl):
        z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
        #print(z_line)
        SNR_line=np.zeros(len(z_line))
        
        
        print(L,l)
        print(SNR_L_l(L,l,z1))
        print(SNR_L_l(L,l,z2))

        k=0
        for k in range(len(z_line)):
            SNR_line[k]=SNR_L_l(L,l,z_line[k])
        cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
        f=lambda x:cubic_interp(x)-SNR_threshold

        zero_point, _ = brentq(f,z1+0.1*z1,z_line[-1]-0.01, full_output=True)
        

        if zero_point > 0:
            zseq[i][j]=zero_point
            print('done')
        j=j+1
    np.savetxt("/home/ljq/code/MOO/results/redshift/EMRI/redshift_EMRI.txt",zseq,fmt="%50.50f",delimiter=" ")
    j=0
    i=i+1
np.savetxt("/home/ljq/code/MOO/results/redshift/EMRI/redshift_EMRI.txt",zseq,fmt="%50.50f",delimiter=" ")


