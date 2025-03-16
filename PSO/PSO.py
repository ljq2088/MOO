import sys
sys.path.append('/home/ljq/code/MOO')
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
sys.path.append('/public/home/boxuange/ljq/MOO')
#from Sci_Obj.SNR_EMRI import *
from Sci_Obj.SNR_binary import *
#from Sci_Obj.redshift_EMRI import *
def combined_func(x,PSD,**kwargs):
    return -SNR_binary_2(x,PSD)
#PSD=PSD_Sx
PSD=PSD_L_lambda
def crcbpso_test_func_mod(x,params):
    

    if len(x.shape) == 1:
        x=x.reshape(1,-1)
        n_row = 1
        fit_val=np.zeros(n_row)
        valid_pts=np.ones(n_row)
        valid_pts=crcb_chk_std_srch_rng(x)
        fit_val[~valid_pts]=np.inf
        x[valid_pts]=s2rv(x[valid_pts],params)

        for lpc in range(0,n_row):
            if valid_pts[lpc]:
                x_temp=x[lpc]
                fit_val[lpc]=combined_func(x_temp,PSD,figure_file=None)
    else:
        n_row, _ = x.shape
    
        fit_val=np.zeros(n_row)
        valid_pts=np.ones(n_row)
        valid_pts=crcb_chk_std_srch_rng(x)
        fit_val[~valid_pts]=np.inf

        x[valid_pts,:]=s2rv(x[valid_pts,:],params)

        for lpc in range(0,n_row):
            if valid_pts[lpc]:
                x_temp=x[lpc,:]
                fit_val[lpc]=combined_func(x_temp,PSD,figure_file=None)
    return fit_val,x,r2sv(x,params)
#Test crcbpso
#7个参数的取值范围
# ffparams = {
#    'r_min':np.array([5.0e6,25.0e6,59.0e6,57.0e6,3.363e9,0.44,2.06e-3,1.0*10**8,200.0]),
#    'r_max':np.array([5.0e6,25.0e6,59.0e6,57.0e6,3.363e9,0.44,2.06e-3,10.0*10**9,1200.0]) }

ffparams = {
   'r_min':np.array([1.0*10**8,200.0]),
   'r_max':np.array([10.0*10**9,1200.0]) }
#Fitness function handle
fit_func_handle=lambda x: crcbpso_test_func_mod(x,ffparams) 
#Call PSO
#np.random.seed(0)#Ensure that the random seed is correct
pso_out=crcbpso(fit_func_handle,2)
#Estimated parameters
std_coord=pso_out['best_location']
_,real_coord,_=fit_func_handle(std_coord)
print(f"Best location: {real_coord}")
print(f"Best fitness: {pso_out['best_fitness']}")