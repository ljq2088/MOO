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



#准备波形数据
Mpc=config.Mpc
fmin=config.fmin
Deff = 2 * 1e3 * Mpc
beta = 6
m1 = config.m1
m2 = config.m2

M_tot = (m1 + m2)  # M_tot in kilograms
eta = (m1*m2)/(M_tot**2)  # Symmetric mass ratio [dimensionless]=
M_chirp = M_tot*eta**(3/5)  # Chirp mass in units of kilograms 


f_max = final_frequency(M_chirp,eta)  # Calculate maximum frequency (Schwarzschild ISCO frequency)
t_max = T_chirp(fmin,M_chirp,eta)     # Calculate maximum chirping time that binary radiates stuff

logMchirp = np.log(M_chirp)
pars = [logMchirp,eta,beta,Deff] # array of parameters for waveform.
eps_GR = 0
eps_AP = (4*1*1e-2)

delta_t = 1/(2*f_max) # Delta
t = np.arange(0,t_max,delta_t) 
n_t = len(t)  

delta_f = 1/(n_t*delta_t)         # Extract sampling frequency

freq_bin = np.arange(fmin,f_max,delta_f)     # Extract frequency series
n_f = len(freq_bin)

h_f = htilde(freq_bin,eps_GR,pars)

#确定参数范围计算SNR
L1=5.0*10**8
L2=10.0*10**9
dL=1.0*10**8
l1=200.0
l2=1600.0
dl=50.0
m=0
n=0
L=L1
l=l1
while L<L2:
    L=L+dL
    m+=1
while l<l2:
    l=l+dl
    n+=1
L=L1
l=l1
df=freq_bin[1]-freq_bin[0]
SNR=np.zeros((m,n))