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

from MOO_func import tukey
from MOO_func import inner_prod
from MOO_func import PSD_Lisa
from MOO_func import PSD_Taiji
from MOO_func import PSD_Tianqin
from MOO_func import PSD_armlength_dependent
from MOO_func import Find_zero
from MOO_func import E
from MOO_func import DL
from MOO_func import PSD_L_lambda
from MOO_func import SNR_for_diff_para
from MOO_func import SNR_M_D_L_l
from MOO_func import htilde
from MOO_func import T_chirp
from MOO_func import final_frequency
from MOO_func import SNR_binary_redshift
from scipy.optimize import brentq
from MOO_func import secant_method
use_gpu = False

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

# set omp threads one of two ways
num_threads = 4

# this is the general way to set it for all computations
# from few.utils.utility import omp_set_num_threads
# omp_set_num_threads(num_threads)

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    num_threads=num_threads,  # 2nd way for specific classes
)

gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform")

# parameters
T = 1 # years
dt = 10  # seconds
M = 5e5
a = 0.98
mu = 50
p0 = 11.0
e0 = 0.1
x0 = 0.7  # notice this is x_I, not Y. The AAK waveform can convert to Y. 
qK = 0.2  # polar spin angle
phiK = 0.2  # azimuthal viewing angle
qS = 0.3  # polar sky angle
phiS = 0.3  # azimuthal viewing angle
dist = 10# distance
Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

# h = gen_wave(
#     M,
#     mu,
#     a,
#     p0,
#     e0,
#     x0,
#     dist,
#     qS,
#     phiS,
#     qK,
#     phiK,
#     Phi_phi0,
#     Phi_theta0,
#     Phi_r0,
#     T=T,
#     dt=dt,
# )

# plt.plot(h.real[:2000])
# plt.show

# # temp=h[:2000]
# # h=temp
# print(len(h))

# para=np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T,dt])
# print(para)
# temp=str(para)
# with open('parametersAAK_PN5.txt', 'w') as f:
#     f.write(temp)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from math import pi as Pi



# para=np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T,dt])
# temp=para
# #para=[float(x) for x in temp]
# # temp=h[-10000:]
# # h=temp
# wave1 = np.array(h)
# f = np.array(np.arange(len(h))/dt/ len(h))
# plt.loglog(f,wave1.real,'ro')
# plt.show()

# tukey_seq=[tukey(i,len(h),1/8) for i in range(0,len(h))]
# wave1 = tukey_seq*wave1

# waveform1 = fft(wave1)
# waveform2 = np.column_stack((waveform1, f))
# temp=waveform2.real*waveform2.real+waveform2.imag*waveform2.imag
# waveform = np.sqrt(temp)


# plt.loglog(f, np.abs(waveform1), 'ro')
# plt.show()

# np.savetxt("waveform.txt", waveform, fmt="%50.50f", delimiter=" ")

# import re
# with open('waveform.txt', 'r') as f:
#     text = f.read()
#     patn = re.sub(r"[\([{})\]]", "", text)

# with open('waveformAAK_PN5.txt', 'w') as f:
#     f.write(patn)

GM_sun = 1.3271244*1e20 #   这个式子等于  G * M_sun
c =2.9979246*1e8
M_sun =1.9884099*1e30
G = 6.6743*1e-11
pc= 3.0856776*1e16

Mpc = (10**6) * pc


#选择一个质量计算探测距离随臂长关系
M_final=10**5
L1=5*10**8
L2=10*10**9
dL=1*10**8
l1=500
l2=1200
dl=100
t0 =1.
phi0 =0.
fmin = 1e-4

# variables to sample through

m1 = 1e5  
m2 = 2*1e5

state1=[0]
np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")
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

L1=5.0*10**8
L2=10.0*10**9
dL=5.0*10**8
l1=200.0
l2=1200.0
dl=100.0
# SNRlist2=SNR_for_diff_para(freq_bin,h_f,PSD_L_lambda,L1,L2,dL,l1,l2,dl,figure_file=None)
from scipy import interpolate
# np.savetxt("SNR_binary.txt",SNRlist2,fmt="%50.50f",delimiter=" ")
z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
#SNR_line=np.zeros(len(z_line))
zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))

SNR_threshold=100
zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))

L=np.arange(L1,L2,dL)

l=np.arange(l1,l2,dl)


for i in np.arange(len(L)):
    for j in np.arange(len(l)):
        print(L[i],l[j])
        print(SNR_binary_redshift(m1,m2,L[i],l[j],fmin,z1))
        print(SNR_binary_redshift(m1,m2,L[i],l[j],fmin,z2))
        SNR_line=np.zeros(len(z_line))
        k=0
        for k in range(len(z_line)):
            SNR_line[k]=SNR_binary_redshift(m1,m2,L[i],l[j],fmin,z_line[k])
        cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
        f=lambda x:SNR_binary_redshift(m1,m2,L[i],l[j],fmin,x)-SNR_threshold
        print(f(z1+0.1*z1),f(z2-0.1*z2))

        zeropoint=brentq(f,z1+0.1*z1,z2-0.1*z2)
        
        
        
        if zeropoint > 0:
            zseq[i][j]=zeropoint
            print('done')
        
    np.savetxt("/home/ljq/code/Multi-Obj-Opt2.0/results/redshift/binary/redshift_binary_2.txt",zseq,fmt="%50.50f",delimiter=" ")
np.savetxt("/home/ljq/code/Multi-Obj-Opt2.0/results/redshift/binary/redshift_binary_2.txt",zseq,fmt="%50.50f",delimiter=" ")
state1=[1]
np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")


# L1=2*10**9
# L2=4*10**9
# dL=1*10**8
# l1=200
# l2=1800
# dl=50
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_binary_redshift(m1,m2,L,l,fmin,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift_binary2.txt",zseq,fmt="%50.50f",delimiter=" ")
# state1=[2]
# np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")


# L1=4*10**9
# L2=6*10**9
# dL=1*10**8
# l1=200
# l2=1800
# dl=50
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_binary_redshift(m1,m2,L,l,fmin,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift_binary3.txt",zseq,fmt="%50.50f",delimiter=" ")
# state1=[3]
# np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")


# L1=6*10**9
# L2=8*10**9
# dL=1*10**8
# l1=200
# l2=1800
# dl=50
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_binary_redshift(m1,m2,L,l,fmin,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift_binary4.txt",zseq,fmt="%50.50f",delimiter=" ")
# state1=[4]
# np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")


# L1=8*10**9
# L2=10*10**9
# dL=1*10**8
# l1=200
# l2=1800
# dl=50
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_binary_redshift(m1,m2,L,l,fmin,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift_binary5.txt",zseq,fmt="%50.50f",delimiter=" ")
# state1=[5]
# np.savetxt("state1.txt",state1,fmt="%50.50f",delimiter=" ")


# redshift1=np.loadtxt(r'redshift_binary1.txt')
# redshift2=np.loadtxt(r'redshift_binary2.txt')
# redshift3=np.loadtxt(r'redshift_binary3.txt')
# redshift4=np.loadtxt(r'redshift_binary4.txt')
# redshift5=np.loadtxt(r'redshift_binary5.txt')
# redshift=np.row_stack((redshift1,redshift2,redshift3,redshift4,redshift5))
# np.savetxt("redshift_binary.txt",redshift,fmt="%50.50f",delimiter=" ")