from para_rb import *
import sys
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from math import pi as Pi
from para_rb import *
from scipy import interpolate



for i in np.arange(len(L)):
    for j in np.arange(len(l)):
        print(L[i],l[j])
        print(SNR_binary_redshift(L[i],l[j],z1))
        print(SNR_binary_redshift(L[i],l[j],z2))
        SNR_line=np.zeros(len(z_line))
        k=0
        for k in range(len(z_line)):
            SNR_line[k]=SNR_binary_redshift(L[i],l[j],z_line[k])
        cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
        f=lambda x:SNR_binary_redshift(L[i],l[j],x)-SNR_threshold
        print(f(z1+0.1*z1),f(z2-0.1*z2))

        zeropoint,_=brentq(f,z1+0.1*z1,z2-0.1*z2,full_output=True)
        
        
        
        if zeropoint > 0:
            zseq[i][j]=zeropoint
            print('done')
        
    np.savetxt("/home/ljq/code/MOO/results/redshift/binary/redshift_binary_3.txt",zseq,fmt="%50.50f",delimiter=" ")
np.savetxt("/home/ljq/code/MOO/results/redshift/binary/redshift_binary_3.txt",zseq,fmt="%50.50f",delimiter=" ")
