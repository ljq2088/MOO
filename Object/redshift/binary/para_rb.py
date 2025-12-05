import sys
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import sys
sys.path.append('/home/ljq/code/MOO')
from config.config import config
from waveform.binary_waveform import *
from utils.inner_prod import inner_prod
from utils.PSD import PSD_L_lambda
from utils.Lumi_redshift import *
from utils.PSD import *
from utils.inner_prod import *
from utils.zeropoint import *
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from config.config import config



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

#准备波形数据
L1=5.0*10**8
L2=10.0*10**9
dL=1.0*10**8
l1=200.0
l2=1600.0
dl=20.0

itr=100000
e=1
# z1=0.000001
# z2=100
z1=0.1
z2=20
dz=0.01
log_z1=np.log(z1)
log_z2=np.log(z2)
dlog_z=0.5

#插值红移序列
z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
#存储红移的矩阵
zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
#阈值
SNR_threshold=config.SNR_threshold_binary

#参数序列
L=np.arange(L1,L2,dL)
l=np.arange(l1,l2,dl)


def SNR_binary_redshift(L,l,z):
    pc= 3.0856776*1e16
    Gpc = (10**9) * pc
    
    
    
    Deff = DL(z)*Gpc
    beta = 6
    
    M_tot = (m1 + m2)  # Total mass
    eta = (m1*m2)/(M_tot**2)  # Symmetric mass ratio [dimensionless]=
    M_chirp = M_tot*eta**(3/5)  # Chirp mass in units of kilograms 
    f_max = final_frequency(M_chirp,eta)  # Calculate maximum frequency (Schwarzschild ISCO frequency)
    t_max = T_chirp(fmin,M_chirp,eta)     # Calculate maximum chirping time that binary radiates stuff
    logMchirp = np.log(M_chirp)
    f_max = final_frequency(M_chirp,eta)  # Calculate maximum frequency (Schwarzschild ISCO frequency)
    t_max = T_chirp(fmin,M_chirp,eta) 

    pars = [logMchirp,eta,beta,Deff]
    delta_t = 1/(2*f_max)         # Set sampling interval so that we can resolved frequencies of BOTH signals
    t = np.arange(0,t_max,delta_t)     
    n_t = len(t)                      # Extract length
    delta_f = 1/(n_t*delta_t)         # Extract sampling frequency
    freq_bin = np.arange(fmin,f_max,delta_f)     # Extract frequency series
    
    if np.any(PSD_L_lambda(freq_bin,[L,l]) <= 0):
        print("序列中有小于零的值")
    
    
    n_f = len(freq_bin)       
    h=htilde(freq_bin,eps_GR,pars)
    PSD=PSD_L_lambda(freq_bin,[L,l])

  






# 检查 h 和 PSD 是否包含无效的浮点值
    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
        print("h 包含无效的浮点值")
    if np.any(np.isnan(PSD)) or np.any(np.isinf(PSD)):
        print("PSD 包含无效的浮点值")

    # 检查 PSD 中是否有零值
    if np.any(PSD == 0):
        print("PSD 包含零值，可能导致除以零错误")

    SNR2=(4*delta_f)*np.real(sum(h*np.conjugate(h)/PSD))
    if SNR2 < 0 or np.isnan(SNR2) or np.isinf(SNR2):
        print(f"Invalid SNR2 value: {SNR2} for L: {L}, l: {l}")
    
    return np.sqrt(SNR2) #

