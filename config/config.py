import numpy as np
class config:
    GM_sun = 1.3271244*1e20 #   这个式子等于  G * M_sun
    c =2.9979246*1e8
    M_sun =1.9884099*1e30
    G = 6.6743*1e-11
    pc= 3.0856776*1e16
    Mpc = (10**6) * pc
    e = 1.6e-19   #库伦

    indicators=np.array([1.7*10**8,2.5*10**9,3*10**9])

    H0 = 67.4  #（哈勃常数） 
    Ωm = 0.315 #（物质密度参数） 
    ΩΛ = 0.685 #（暗能密度参数）
    c0=299792.458 #（光速）

    M_final=10.0**5
    t0 =1.
    phi0 =0.
    fmin = 1e-4
    #参数范围
    L1=1.0*10**8
    L2=10.0*10**9
    dL=1.0*10**8
    l1=200.0
    l2=1200.0
    dl=50.0

    SNR_threshold_EMRI=10.0
    SNR_threshold_binary=100.0

    m1=1e5
    m2=2*1e5


  
    

