import numpy as np
def inner_prod(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """
    return (2*1/len(sig1_f)**2/delta_f)*np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD))