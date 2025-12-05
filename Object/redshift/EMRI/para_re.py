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
from waveform.EMRI_waveform import *



#参数范围
L1=5.0*10**8
L2=10.0*10**9
dL=1.0*10**8
l1=200.0
l2=1600.0
dl=20.0
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

SNR_threshold=config.SNR_threshold_EMRI

def SNR_L_l(L,l,z):
    
    
    q=1e-4#mass ratio
    
    
    # parameters
    M = M_final
    mu=M_final*q
    p0 = 12.0
    e0 = 0.4
    theta = np.pi/3  # polar viewing angle
    phi = np.pi/4  # azimuthal viewing angle
    dt = 10
    dist=DL(z)
    T=0.5
    h = few_wf(M, mu, p0, e0, theta, phi,dist=dist, dt=dt, T=T)  
    h=h.get()

    wave1 = np.array(h)
    f = np.array(np.arange(len(h))/dt/ len(h))  
    #tukey_seq=np.array([tukey(i,len(h),1/8) for i in range(0,len(h))])
    #print(len(wave1),len(tukey_seq))
    wave1 *= tukey(len(h), 1/8)
    waveform1 = fft(wave1)
    waveform2 = np.column_stack((waveform1, f))
    temp=waveform2.real*waveform2.real+waveform2.imag*waveform2.imag
    waveform = np.sqrt(temp)

    fseq=waveform[1:len(waveform[:,1]),1]
    h_f=waveform[1:len(waveform[:,0]),0]
    df=waveform[1,1]-waveform[0,1]

    # PSD=PSD_Lisa(fseq)
    PSD=PSD_L_lambda(fseq,[L,l])
    SNR2 = inner_prod(h_f,h_f,PSD,df)
    return np.sqrt(SNR2)

#准备插值的红移序列寻找阈值临界点
z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
SNR_line=np.zeros(len(z_line))
zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))