import os
import sys
sys.path.append('/home/ljq/code/MOO')
sys.path.append('/home/ljq/code/MOO/PSO')
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from PSO_func import *
from utils.inner_prod import *
from utils.Lumi_redshift import *  
from utils.PSD import *
from utils.zeropoint import *
from waveform.binary_waveform import *
from waveform.EMRI_waveform import *
from scipy.fftpack import fft, ifft,fftfreq
import math
from scipy.signal.windows import tukey
from config.config import config
file_saved_path = "/home/ljq/code/MOO/waveform/waveformFEW.txt"

# 检查文件是否存在
if not os.path.exists(file_saved_path):
    # 如果文件不存在，生成并保存数据
    q=1e-4#mass ratio
    # parameters
    M = config.M_final
    mu=M*q
    p0 = 12.0
    e0 = 0.4
    theta = np.pi/3  # polar viewing angle
    phi = np.pi/4  # azimuthal viewing angle
    dt = 10
    dist=1
    T=0.5
    h = few_wf(M, mu, p0, e0, theta, phi,dist=dist, dt=dt, T=T)  
    h=h.get()
    wave1 = np.array(h)
    f = np.array(np.arange(len(h))/dt/ len(h))
    tukey_seq=[tukey(i,len(h),1/8) for i in range(0,len(h))]
    wave1 = tukey_seq*wave1

    waveform1 = fft(wave1)
    waveform2 = np.column_stack((waveform1, f))
    temp=waveform2.real*waveform2.real+waveform2.imag*waveform2.imag
    waveform = np.sqrt(temp)
    np.savetxt(file_saved_path, waveform, fmt="%50.50f", delimiter=" ")
    print(f"文件 {file_saved_path} 已生成。")
    data = np.loadtxt(file_saved_path)
else:
    # 如果文件存在，直接读取
    data = np.loadtxt(file_saved_path)
    print(f"文件 {file_saved_path} 已存在，数据已加载。")

freq_SNR_EMRI=data[1:len(data[:,1]),1]
h_SNR_EMRI=data[1:len(data[:,0]),0]

print(freq_SNR_EMRI.shape,h_SNR_EMRI.shape)
def SNR_EMRI_1(paras,PSD,figure_file=None,**kwargs):

    L=paras[0]
    l=paras[1]
    
    df=freq_SNR_EMRI[1]-freq_SNR_EMRI[0]
   
    PSD_seq=PSD(freq_SNR_EMRI,L,l)
    SNRtemp = inner_prod(h_SNR_EMRI,h_SNR_EMRI,PSD_seq,df)
    SNR=np.sqrt(SNRtemp)
            
            
  
    return SNR
