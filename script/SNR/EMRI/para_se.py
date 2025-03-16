import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from config.config import config

from utils.inner_prod import inner_prod
from utils.PSD import PSD_L_lambda
#准备波形数据
data=np.loadtxt(r'/home/ljq/code/MOO/Workspace/waveformAAK_PN5.txt')
freq=data[1:len(data[:,1]),1]
h_local=data[1:len(data[:,0]),0]
L1=1*10**8
L2=10*10**9
dL=1*10**8
l1=200
l2=1200
dl=50
#构造二维矩阵存储SNR
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
df=freq[1]-freq[0]
SNR=np.zeros((m,n))