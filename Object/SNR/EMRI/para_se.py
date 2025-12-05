import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from config.config import config

from utils.inner_prod import inner_prod
from utils.PSD import PSD_L_lambda
#准备波形数据
loaded_data = np.load("/home/ljq/code/MOO/Object/SNR/EMRI/FEW.npz")
freq = loaded_data["freq"]
h_local = loaded_data["h_local"]

L1=5.0*10**8
L2=10.0*10**9
dL=1.0*10**8
l1=200.0
l2=1600.0
dl=20.0
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