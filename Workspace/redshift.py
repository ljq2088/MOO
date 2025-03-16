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
from scipy.optimize import brentq
use_gpu = False

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



# # parameters
# M = 5e5
# mu = 50
# p0 = 12.0
# e0 = 0.4
# theta = np.pi/3  # polar viewing angle
# phi = np.pi/4  # azimuthal viewing angle
# dt = 10
# dist=1
# T=0.5

# h = few(M, mu, p0, e0, theta, phi,dist=dist, dt=dt, T=T)  


# # plt.plot(h.real[:2000])
# # plt.show

# # temp=h[:2000]
# # h=temp
# print(len(h))

# # para=np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T,dt])
# # print(para)
# # temp=str(para)
# # with open('parametersAAK_PN5.txt', 'w') as f:
# #     f.write(temp)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft,fftfreq
import math
from math import pi as Pi



# para=np.array([M,mu,a,p0,e0,x0,dist,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,T,dt])
# temp=para
#para=[float(x) for x in temp]
# temp=h[-10000:]
# h=temp
# h=h.get()
# wave1 = np.array(h)
# f = np.array(np.arange(len(h))/dt/ len(h))
# # plt.loglog(f,wave1.real,'ro')
# # plt.show()

# tukey_seq=np.array([tukey(i,len(h),1/8) for i in range(0,len(h))])
# wave1 = tukey_seq*wave1

# waveform1 = fft(wave1)
# waveform2 = np.column_stack((waveform1, f))
# temp=waveform2.real*waveform2.real+waveform2.imag*waveform2.imag
# waveform = np.sqrt(temp)

#print(1)
# plt.loglog(f, np.abs(waveform1), 'ro')
# plt.show()

# np.savetxt("waveform.txt", waveform, fmt="%50.50f", delimiter=" ")

# import re
# with open('waveform.txt', 'r') as f:
#     text = f.read()
#     patn = re.sub(r"[\([{})\]]", "", text)

# with open('waveformAAK_PN5.txt', 'w') as f:
#     f.write(patn)
state=[0]
np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")
#选择一个质量计算探测距离随臂长关系
M_final=10**5
L1=5.0*10**8
L2=10.0*10**9
dL=5.0*10**8
l1=200.0
l2=1200.0
dl=100.0
#test
# L1=18*10**8
# L2=2.5*10**9
# dL=1*10**8
# l1=200
# l2=1200
# dl=50

itr=100000
e=0.1
z1=0.001
z2=4
dz=0.01
log_z1=np.log(z1)
log_z2=np.log(z2)
dlog_z=0.5

print(SNR_M_D_L_l(M_final,L1,l1,4))
SNR_threshold=10


from scipy import interpolate




i=0
j=0

import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from multiprocessing import Pool, cpu_count
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# def compute_SNR(L, l, z_line, M_final, SNR_threshold):
#     SNR_line = np.zeros(len(z_line))
    
#     for k in range(len(z_line)):
#         SNR_line[k] = SNR_M_D_L_l(M_final, L, l, z_line[k])
        
#     cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
#     f = lambda x: cubic_interp(x) - SNR_threshold
    
#     try:
#         zeropoint = brentq(f, z_line[0] + 0.1 * z_line[0], z_line[-1] - 0.01)
#     except ValueError:
#         zeropoint = np.nan  # 如果找不到零点，就返回 NaN
    
#     return zeropoint

# def parallel_compute_SNR(args):
#     L, l, z_line, M_final, SNR_threshold = args
#     return compute_SNR(L, l, z_line, M_final, SNR_threshold)

z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
SNR_line=np.zeros(len(z_line))
zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# params = [(L, l, z_line, M_final, SNR_threshold) for L in np.arange(L1, L2, dL) for l in np.arange(l1, l2, dl)]
# with Pool(cpu_count()) as pool:
#     results = pool.map(parallel_compute_SNR, params)

# i = 0
# for L in np.arange(L1, L2, dL):
#     j = 0
#     for l in np.arange(l1, l2, dl):
#         zseq[i][j] = results[i * len(np.arange(l1, l2, dl)) + j]
#         j += 1
#     i += 1
# np.savetxt("redshift.txt",zseq,fmt="%50.50f",delimiter=" ")



for L in np.arange(L1,L2,dL):
    for l in np.arange(l1,l2,dl):
        z_line=np.exp(np.arange(log_z1,log_z2,dlog_z))
        #print(z_line)
        SNR_line=np.zeros(len(z_line))
        
        
        print(L,l)
        print(SNR_M_D_L_l(M_final,L,l,z1))
        print(SNR_M_D_L_l(M_final,L,l,z2))

        k=0
        for k in range(len(z_line)):
            SNR_line[k]=SNR_M_D_L_l(M_final,L,l,z_line[k])
        cubic_interp = interpolate.interp1d(z_line, SNR_line, kind='cubic')
        f=lambda x:cubic_interp(x)-SNR_threshold

        zeropoint=brentq(f,z1+0.1*z1,z_line[-1]-0.01)
        

        if zeropoint > 0:
            zseq[i][j]=zeropoint
            print('done')
        j=j+1
    np.savetxt("/home/ljq/code/Multi-Obj-Opt2.0/results/redshift/EMRI/redshift.txt",zseq,fmt="%50.50f",delimiter=" ")
    j=0
    i=i+1
np.savetxt("/home/ljq/code/Multi-Obj-Opt2.0/results/redshift/EMRI/redshift.txt",zseq,fmt="%50.50f",delimiter=" ")
state=[1]
np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")


# L1=6.0*10**9
# L2=8.0*10**9
# dL=1.0*10**8
# l1=200.0
# l2=1200.0
# dl=50.0
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_M_D_L_l(M_final,L,l,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift4.txt",zseq,fmt="%50.50f",delimiter=" ")
# state=[2]
# np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")

# L1=4.0*10**9
# L2=6.0*10**9
# dL=1.0*10**8
# l1=200.0
# l2=1200.0
# dl=50.0
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_M_D_L_l(M_final,L,l,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift3.txt",zseq,fmt="%50.50f",delimiter=" ")
# state=[3]
# np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")

# L1=2.0*10**9
# L2=4.0*10**9
# dL=1.0*10**8
# l1=200.0
# l2=1200.0
# dl=50.0
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_M_D_L_l(M_final,L,l,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift2.txt",zseq,fmt="%50.50f",delimiter=" ")
# state=[4]
# np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")

# L1=1.0*10**8
# L2=2.0*10**9
# dL=1.0*10**8
# l1=200.0
# l2=1200.0
# dl=50.0
# SNR_threshold=10
# zseq=np.zeros((len(np.arange(L1,L2,dL)),len(np.arange(l1,l2,dl))))
# i=0
# j=0
# for L in np.arange(L1,L2,dL):
#     for l in np.arange(l1,l2,dl):
#         f=lambda x:SNR_M_D_L_l(M_final,L,l,x)-SNR_threshold
#         zeropoint=brentq(f,z1,z2)
#         if zeropoint>0:
#             zseq[i][j]=zeropoint
#         j=j+1
#     j=0
#     i=i+1
# np.savetxt("redshift1.txt",zseq,fmt="%50.50f",delimiter=" ")
# state=[5]
# np.savetxt("state.txt",state,fmt="%50.50f",delimiter=" ")

# redshift1=np.loadtxt(r'redshift1.txt')
# redshift2=np.loadtxt(r'redshift2.txt')
# redshift3=np.loadtxt(r'redshift3.txt')
# redshift4=np.loadtxt(r'redshift4.txt')
# redshift5=np.loadtxt(r'redshift5.txt')
# redshift=np.row_stack((redshift1,redshift2,redshift3,redshift4,redshift5))
# np.savetxt("redshift.txt",redshift,fmt="%50.50f",delimiter=" ")
