from config.config import config
from scipy import integrate
import numpy as np
import math
c0=config.c0
H0=config.H0
Ωm=config.Ωm
ΩΛ=config.ΩΛ


def E(z):
    return 1/np.sqrt(Ωm*(1+z)**3+ΩΛ)

def DL(z):
    v,err=integrate.quad(E,0,z)
    return (1+z)*c0/H0*v/1000
