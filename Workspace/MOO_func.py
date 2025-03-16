#Basic functions for multi-objective optimization


import sys
import os

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap, 
                               get_mismatch, 
                               get_fundamental_frequencies, 
                               get_separatrix, 
                               get_mu_at_t, 
                               get_p_at_t, 
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase
use_gpu = True

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

import numpy as np
import math
#常数
from math import pi 
GM_sun = 1.3271244*1e20 #   这个式子等于  G * M_sun
c =2.9979246*1e8
M_sun =1.9884099*1e30
G = 6.6743*1e-11
pc= 3.0856776*1e16

Mpc = (10**6) * pc
#BHB waveform
def htilde(f,eps,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    
    # Load the parameters
    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    beta_true = params[2]
    Deff = params[3]
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    
    # Amplitude explicitly given in terms of units of seconds. This is a Continuous 
    # Time Fourier representation of the signal in the time domain.
    
    amplitude_1 = - (Mpc/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 *Mpc))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    
    # Phase: add or remove PN orders here as you see fit.
    t0 =1.
    phi0 =0.
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
    psi1PN = (3715/756 + (55/9)*eta_true)*v**(-3)
    psi1_5PN_tails = -16*pi*v**(-2)
    psi1_5PN_spin = 4*beta_true*v**(-2)
    
    psi2PN = (15293365/508032+(27145/504)*eta_true+(3085/72)*eta_true**2)*v**(-1)
    psi25PNlog = pi*(38645/252- (65/3) *eta_true)* np.log(v)
    psi3PN = v*(11583231236531/4694215680 - (640/3) * (pi**2) -6848/21 *np.euler_gamma
              + eta_true*(-15335597827/3048192 + (2255/12) * (pi**2) - 1760/3 * theta - 12320/9 * delta)
              + (eta_true**2) *76055/1728 - (eta_true**3) * 127825/1296 - 6848/21 * np.log(4))
    psi3PNlog = - 6848/21 *v * np.log(v)
    psi35PN = pi * v**2 * (77096675./254016 + (378515./1512) *eta_true - 74045./756 * (eta_true**2)* (1-eps))
    psi_fullPN = (3/(128*eta_true))*(v**(-5)+psi1PN+psi1_5PN_tails+psi1_5PN_spin+psi2PN
                                  + psi25PNlog + psi3PN + psi3PNlog + psi35PN)
    psi = psi_const + psi_fullPN 
    return amplitude* np.exp(-1j*psi)


def T_chirp(fmin,M_chirp,eta):
    """
    Calculate time elapsed until merger.
    
    """

    #M = (m1 + m2)*M_sun
    M_chirp *= M_sun
    
    M = M_chirp*eta**(-3/5)
    v_low = (pi*G*M_chirp*eta**(-3/5)/(c**3) * fmin)**(1/3)
    
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    gamma = np.euler_gamma
    
    pre_fact = ((5/(256*eta)) * G*M/(c**3))
    first_term = (v_low**(-8) + (743/252 + (11/3) * eta ) * (v_low **(-6)) - (32*pi/5)*v_low**(-5)
                +(3058673/508032 + (5429/504)*eta + (617/72)*eta**2)*v_low**(-4)
                 +(13*eta/3 - 7729/252)*pi*v_low**-3)
    
    second_term = (6848*gamma/105 - 10052469856691/23471078400 + 128*pi**2/3 + (
    3147553127/3048192 - 451*(pi**2)/12)*eta - (15211*eta**2)/1728 + (2555*eta**3 / 1296) +
                   (6848/105)*np.log(4*v_low))*v_low**-2
    
    third_term = ((14809/378)*eta**2 - (75703/756) * eta - 15419335/127008)*pi*v_low**-1
    return pre_fact * (first_term + second_term + third_term)

def final_frequency(M_chirp,eta):
    """
    Schwarzschild ISCO
    """
    M_tot = M_chirp*eta**(-3/5) * M_sun
    
    return (c**3)/(6*np.sqrt(6)*pi*G*M_tot)
    


def inner_prod(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """
    return (2*1/len(sig1_f)**2/delta_f)*np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD))

def PSD_Lisa(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = c/(2*pi*L)   
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Lisa_no_Response(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = c/(2*pi*L)   
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((1/(L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4))) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

def PSD_Taiji(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 3.0*10**9   # Length of Taiji arm
    f0 = c/(2*pi*L)      
    
    Poms = ((8*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Tianqin(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 1.7*10**8   # Length of LISA arm
    f0 = c/(2*pi*L)      
    
    Poms = ((1.0*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (1*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Tianqin_modified(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 1.7*10**8   # Length of LISA arm
    f0 = c/(2*pi*L)      
    
    Poms = ((1.0*10**-12)**2)  # Optical Metrology Sensor
    Pacc = (1*10**-15)**2*(1 + (10**-3/(10*f)))  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = (20/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0))  # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def Index(alpha,beta,f,f_0):
    return alpha*np.power(f/f_0,beta)

def inner_prod_for_Index(sig1_f,sig2_f,Index,df,f_1,f_2,freq):
    freq_seq=[]
    for i in np.arange(0,len(sig1_f),1):
        if freq[i]>=f_1 and freq[i]<=f_2:
            freq_seq=np.append(freq_seq,i)
            
    fseq=[int(x) for x in freq_seq]    
    
    return (4*df) *np.real(sum(sig1_f[fseq]*np.conjugate(sig2_f[fseq])/Index[fseq]**2))
def inner_prod_for_Index2(sig1_f,sig2_f,Index,df):
        
    return (4*df) *np.real(sum(sig1_f*np.conjugate(sig2_f)/Index**2))

#coefficient
from numpy import polyfit,poly1d
temp=np.array([1.7*10**8,2.5*10**9,3*10**9])
x=temp**2
y=np.array([(1.0*10**-12)**2 ,(1.5*10**-11)**2,(8*10**-12)**2])
coeff=polyfit(x,y,1)

def PSD_armlength_dependent(f,L):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    f0 = c/(2*np.pi*L)      
    Poms_LISA=(1.5*10**-11)**2
    Poms_Tianqin = (1.0*10**-12)**2  
    Poms_Taiji=(8*10**-12)**2
    
    Poms=(coeff[0]*L*L+coeff[1])*(1 + ((2*10**-3)/f)**4)
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
    
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
#PSD depends on arm length and wavelength of laser
def PSD_L_lambda(f,L,Lambda):#[Lambda]=nm
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    f0 = c/(2*pi*L)      
    Poms_LISA=(1.5*10**-11)**2
    Poms_Tianqin = (1.0*10**-12)**2   
    Poms_Taiji=(8*10**-12)**2
    
    Poms=(coeff[0]*L*L+coeff[1])*(1 + ((2*10**-3)/f)**4)*Lambda/1000
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)*(1000/Lambda)**2 # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    # print('freq_bin的长度为',len(f))
    # print(f)
    # print(min(Sc),min(Poms),min(Pacc))
    # print(f"Sc: min={np.min(Sc)}, max={np.max(Sc)}")
    # print(f"Poms: min={np.min(Poms)}, max={np.max(Poms)}")
    # print(f"Pacc: min={np.min(Pacc)}, max={np.max(Pacc)}")
    # print(f"(4 * Pacc) / (np.power(2 * np.pi * f, 4)): {np.min((4 * Pacc) / (np.power(2 * np.pi * f, 4)))}")
    # print(f"1 + 0.6 * (f / f0)**2: {np.min(1 + 0.6 * (f / f0)**2)}")
   
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
    if np.any(PSD<0):
        print('PSD<0',f[PSD<0],((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f[PSD<0],4)))*(1 + 0.6*(f[PSD<0]/f0)*(f[PSD<0]/f0)) + Sc))
    if np.any(np.isinf(PSD)):
        print("Overflow detected in data:", PSD[np.isinf(PSD)])
    if np.any(np.isnan(PSD)):
        print("NaN detected in data:", PSD[np.isnan(PSD)])
    #print(min(PSD))    
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD


#二分法求零点
def Find_zero(f, x1, x2, e, iter):
    if f(x1) * f(x2) >= 0:
        return -1

    if f(x1) > 0:
        x1, x2 = x2, x1

    for _ in range(iter):
        mid = (x1 + x2) / 2
        val = f(mid)

        if abs(val) < e:
            return mid

        if val < 0:
            x1 = mid
        else:
            x2 = mid

    return (x1 + x2) / 2

#Secant method for zeropoint
def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    """使用割线法查找函数f(x)=0的零点。

    参数:
    f -- 目标函数
    x0, x1 -- 初始两个估计值
    tol -- 容忍误差，当函数值小于此值时停止迭代
    max_iter -- 最大迭代次数

    返回:
    零点的近似值
    """
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        
        # 计算割线的斜率
        if (fx1 - fx0) == 0:
            return -1  # 防止除以零
        slope = (x1 - x0) / (fx1 - fx0)

        # 更新x1和x0
        x0=x1
        x1=max(x1 - fx1 * slope,0.00001)#要求x>1
       
        # 检查是否达到容忍误差
        if abs(f(x1)) < tol:
            return x1
    return -1  # 如果没有在max_iter迭代次数内找到解，则返回None


#Redshift luminosity distance conversion 红移光度距离换算
H0 = 67.4  #（哈勃常数） 
Ωm = 0.315 #（物质密度参数） 
ΩΛ = 0.685 #（暗能密度参数）
c0=299792.458
from scipy import integrate
import math
def E(z):
    return 1/np.sqrt(Ωm*(1+z)**3+ΩΛ)

def DL(z):
    v,err=integrate.quad(E,0,z)
    return (1+z)*c0/H0*v/1000

# def tukey(n, N, a):
    
#     if not (0 <= a <= 1):
#         raise ValueError("Parameter 'a' must be between 0 and 1")
    
#     #print(f"n={n}, N={N}, a={a}")  # 添加调试信息

#     if n == 0 or n == N - 1:
#         return 0
#     elif n < a * (N - 1) / 2:
#         return 0.5 + 0.5 * math.cos(pi * (2 * n / (a * N - a)))
#     elif n >= a * (N - 1) / 2 and n < (N - 1) * (1 - a/2):
#         return 1
#     elif n >= (N - 1) * (1 - a/2) and n <= N - 1:
#         return 0.5 + 0.5 * math.cos(pi * (2 * n / (a * N - a) - 2 / a + 1))
#     else:
#         return 0

#SNR for different parameters 计算不同参数下的信噪比

def SNR_for_diff_para(f,h,PSD,L1,L2,dL,l1,l2,dl,figure_file=None):
    """ 
    Return 2D array of SNR for different parameters

    Parameters:
    -----------
    Armlength: L
    L1: float
        lower bound for L
    L2: float
        upper bound for L
    dL: float
        step size for L
    wavelength: l
    l1: float
        lower bound for l
    l2: float            
        upper bound for l    
    dl: float
        step size for l
    """
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
    df=f[1]-f[0]
    SNR=np.zeros((m,n))
    for i in np.arange(0,m):
        for j in np.arange(0,n):
            PSD_seq=PSD(f,L,l)
            SNRtemp = inner_prod(h,h,PSD_seq,df)
            SNR[i][j]=np.sqrt(SNRtemp)
            
            l=l+dl
        l=l1
        L=L+dL    
  
    return SNR
# T = 4  # years
# dt = 10.0  # seconds
# a = 0.9
# p0 = 11.0
# e0 = 0.2
# x0 = 0.7  # notice this is x_I, not Y. The AAK waveform can convert to Y. 
# qK = 0.2  # polar spin angle
# phiK = 0.2  # azimuthal viewing angle
# qS = 0.3  # polar sky angle
# phiS = 0.3  # azimuthal viewing angle
    
# Phi_phi0 = 1.0
# Phi_theta0 = 2.0
# Phi_r0 = 3.0

gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform")
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from scipy.signal.windows import tukey
#SNR for binary
def SNR_M_D_L_l(M_total,L,l,z):
    
    
    q=1e-4#mass ratio
    
    
    # parameters
    M = M_total
    mu=M_total*q
    p0 = 12.0
    e0 = 0.4
    theta = np.pi/3  # polar viewing angle
    phi = np.pi/4  # azimuthal viewing angle
    dt = 10
    dist=DL(z)
    T=0.5
    h = few(M, mu, p0, e0, theta, phi,dist=dist, dt=dt, T=T)  
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
    PSD=PSD_L_lambda(fseq,L,l)
    SNR2 = inner_prod(h_f,h_f,PSD,df)
    return np.sqrt(SNR2)
eps_GR = 0
eps_AP = (4*1*1e-2)
#SNR criterion for binary redshift calculations
#输入红移z和两个质量m1,m2以及臂长和波长，返回信噪比
def SNR_binary_redshift(m1,m2,L,l,fmin,z):
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
    
    if np.any(PSD_L_lambda(freq_bin,L,l) <= 0):
        print("序列中有小于零的值")
    
    
    n_f = len(freq_bin)       
    h=htilde(freq_bin,eps_GR,pars)
    PSD=PSD_L_lambda(freq_bin,L,l)

    # plt.figure(figsize=(10, 6))
    # plt.loglog(freq_bin, PSD, label="PSD Curve")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("PSD")
    # plt.title("Power Spectral Density (PSD)")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.legend()
    # plt.show()






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