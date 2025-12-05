import sys
import os

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

freq_SNR_binary = np.arange(fmin,f_max,delta_f)     # Extract frequency series
n_f = len(freq_SNR_binary)

h_SNR_binary = htilde(freq_SNR_binary,eps_GR,pars)

def SNR_binary_1(paras,PSD,**kwargs):
    
    
    df=freq_SNR_binary[1]-freq_SNR_binary[0]
   
    PSD_seq=PSD(freq_SNR_binary,paras)
    SNRtemp = inner_prod(h_SNR_binary,h_SNR_binary,PSD_seq,df)
    SNR=np.sqrt(SNRtemp)
            
            
  
    return SNR

# def SNR_binary_1(paras,PSD,**kwargs):

    
#     df=freq_SNR_binary[1]-freq_SNR_binary[0]
   
#     PSD_seq=PSD(freq_SNR_binary,paras)
#     SNRtemp = inner_prod(h_SNR_binary,h_SNR_binary,PSD_seq,df)
#     SNR=np.sqrt(SNRtemp)

#     return SNR