import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# constants
YEAR   = 3.15581497632e7  # year in seconds
Clight = 299792458.       # speed of light (m/s)

##########################################################
################# Noise Curve Methods ####################
##########################################################

def LoadTransfer(self, file_name):
    self.FLAG_R_APPROX = True
    return
    
def Pn(self, f):
    """
    Caclulate the Strain Power Spectral Density for TianQin
    """
    P_oms = (1e-12)**2  # TianQin OMS noise (approx)
    P_acc = (1e-15)**2 * (1. + (1e-4/f)**2)  # TianQin acceleration noise (approx)
    Pn = (P_oms + 4.*P_acc/(2.*np.pi*f)**4)/self.Larm**2
    return Pn
    
def SnC(self, f):
    """
    TianQin confusion noise is often negligible in the baseline design
    Here we set it to zero or a simple placeholder
    """
    return 0.0*f
    
def Sn(self, f):
    if (self.FLAG_R_APPROX == False):
        R = interpolate.splev(f, self.R_INTERP, der=0)
    else:
        R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
    Sn = self.Pn(f)/R + self.SnC(f)
    return Sn

def SnPn(self, f):
    if (self.FLAG_R_APPROX == False):
        R = interpolate.splev(f, self.R_INTERP, der=0)
    else:
        R = 3./20./(1. + 6./10.*(f/self.fstar)**2)*self.NC
    SnPn = self.Pn(f)/R
    return SnPn

##########################################################
################# TianQin Class ##########################
##########################################################

class TianQin():
    """ 
    TianQin class
    -----------------------
    Handles TianQin's detector noise quantities
    """
    
    def __init__(self, Tobs=5*YEAR, Larm=1.7e8, NC=1, transfer_file='R.txt'):
        """
        Tobs - TianQin observation period (default 5 years)
        Larm - TianQin arm length (~1.7e8 m)
        NC   - Number of data channels
        """
        self.Tobs = Tobs 
        self.NC   = NC 
        self.Larm = Larm
        self.fstar = Clight/(2*np.pi*self.Larm) 
        self.LoadTransfer(transfer_file)

    LoadTransfer = LoadTransfer
    Pn  = Pn
    Sn  = Sn
    SnC = SnC
    SnPn= SnPn
    

def PlotSensitivityCurve(f, Sn, figure_file=None):
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.tight_layout()
    ax.set_xlabel(r'f [Hz]', fontsize=20, labelpad=10)
    ax.set_ylabel(r'Characteristic Strain', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlim(1.0e-5, 1.0e0)
    ax.set_ylim(1.0e-22, 1.0e-15)
    ax.loglog(f, np.sqrt(f*Sn))
    plt.show()
    if (figure_file != None):
        plt.savefig(figure_file)
    return
