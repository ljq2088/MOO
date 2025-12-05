from scipy.optimize import brentq
import sys
sys.path.append('/home/ljq/code/MOO')
from config.config import config
from utils.zeropoint import *
from utils.Lumi_redshift import *
from utils.PSD import PSD_L_lambda
from utils.inner_prod import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from scipy.signal.windows import tukey
from waveform.binary_waveform import *

#红移范围
z1=0.001
z2=4
dz=0.01

#波形参数
M_final=10**5
itr=100000
e=0.1

log_z1=np.log(z1)
log_z2=np.log(z2)
dlog_z=0.5

SNR_threshold=config.SNR_threshold_binary
import sys
sys.path.append('/home/ljq/code/MOO')
sys.path.append('/home/ljq/code/MOO/Object')
from redshift.binary.para_rb import *

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

from scipy import interpolate




#准备插值的红移序列寻找阈值临界点
# z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
# SNR_line=np.zeros(len(z_line))
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
from scipy import interpolate
def redshift_binary_1(paras,**kwargs):
    L=paras[0]
    l=paras[1]
    z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
    #print(z_line)
    SNR_line=np.zeros(len(z_line))
    
    k=0
    for k in range(len(z_line)):
        SNR_line[k]=SNR_binary_redshift(L,l,z_line[k])
    cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
    f=lambda x:cubic_interp(x)-SNR_threshold

    zero_point, _ = brentq(f,z1+0.1*z1,z_line[-1]-0.01, full_output=True)
    
    z=0
    if zero_point > 0:
        z=zero_point
    else:
        print("No solution found")
    return z