import sys
import os

import matplotlib.pyplot as plt
%matplotlib inline
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
from utils.inner_prod import inner_prod
from utils.PSD import PSD_L_lambda
import sys
import os
from para_se import *


for i in np.arange(0,m):
    for j in np.arange(0,n):
        PSD_seq=PSD_L_lambda(freq,L,l)
        SNRtemp = inner_prod(h_local,h_local,PSD_seq,df)
        SNR[i][j]=np.sqrt(SNRtemp)
        
        l=l+dl
    l=l1
    L=L+dL    

np.savetxt("/home/ljq/code/MOO/results/SNR/EMRI/SNR_EMRI.txt", SNR, fmt="%50.50f", delimiter=" ")

