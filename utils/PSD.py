import numpy as np
from math import pi
from config.config import config
GM_sun = config.GM_sun #   这个式子等于  G * M_sun
c =config.c
M_sun =config.M_sun
G = config.G
pc= config.pc

Mpc = config.Mpc
e=config.e   #库伦
def PSD_Lisa(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = c/(2*pi*L)   
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Lisa_no_Response(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = c/(2*pi*L)   
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((1/(L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4))) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

def PSD_Taiji(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 3.0*10**9   # Length of Taiji arm
    f0 = c/(2*pi*L)      
    
    Poms = ((8*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Tianqin(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 1.7*10**8   # Length of LISA arm
    f0 = c/(2*pi*L)      
    
    Poms = ((1.0*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (1*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def PSD_Tianqin_modified(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 1.7*10**8   # Length of LISA arm
    f0 = c/(2*pi*L)      
    
    Poms = ((1.0*10**-12)**2)  # Optical Metrology Sensor
    Pacc = (1*10**-15)**2*(1 + (10**-3/(10*f)))  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = (20/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0))  # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
def Index(alpha,beta,f,f_0):
    return alpha*np.power(f/f_0,beta)

def inner_prod_for_Index(sig1_f,sig2_f,Index,df,f_1,f_2,freq):
    freq_seq=[]
    for i in np.arange(0,len(sig1_f),1):
        if freq[i]>=f_1 and freq[i]<=f_2:
            freq_seq=np.append(freq_seq,i)
            
    fseq=[int(x) for x in freq_seq]    
    
    return (4*df) *np.real(sum(sig1_f[fseq]*np.conjugate(sig2_f[fseq])/Index[fseq]**2))
def inner_prod_for_Index2(sig1_f,sig2_f,Index,df):
        
    return (4*df) *np.real(sum(sig1_f*np.conjugate(sig2_f)/Index**2))

#coefficient

from numpy import polyfit,poly1d
temp=config.indicators
x=temp**2
y=np.array([(1.0*10**-12)**2 ,(1.5*10**-11)**2,(8*10**-12)**2])
coeff=polyfit(x,y,1)

def PSD_armlength_dependent(f,L):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    f0 = c/(2*np.pi*L)      
    Poms_LISA=(1.5*10**-11)**2
    Poms_Tianqin = (1.0*10**-12)**2  
    Poms_Taiji=(8*10**-12)**2
    
    Poms=(coeff[0]*L*L+coeff[1])*(1 + ((2*10**-3)/f)**4)
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
    
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD
#PSD depends on arm length and wavelength of laser
def PSD_L_lambda(f,paras):#[Lambda]=nm
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    L=paras[0]
    Lambda=paras[1]
    f0 = c/(2*pi*L)      
    Poms_LISA=(1.5*10**-11)**2
    Poms_Tianqin = (1.0*10**-12)**2   
    Poms_Taiji=(8*10**-12)**2
    
    Poms=(coeff[0]*L*L+coeff[1])*(1 + ((2*10**-3)/f)**4)*Lambda/1000
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)*(1000/Lambda)**2 # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    
   
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
    if np.any(PSD<0):
        print('PSD<0',f[PSD<0],((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f[PSD<0],4)))*(1 + 0.6*(f[PSD<0]/f0)*(f[PSD<0]/f0)) + Sc))
    if np.any(np.isinf(PSD)):
        print("Overflow detected in data:", PSD[np.isinf(PSD)])
    if np.any(np.isnan(PSD)):
        print("NaN detected in data:", PSD[np.isnan(PSD)])
    #print(min(PSD))    
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

#这里给定太极的一些固定参数，如果可以增加LISA的更好
eps_het = 0.8   #Heterodyne interference efficiency
R_pd = 0.68     #A/W    Photodiode responsivity
RIN = 1e-8      #Relative intensity laser noise
N_pd = 4.0      #Photo detector phase number
U_pd = 2.0e-9   #V/sqrt(Hz)  Photodetectorvoltagenoise
C_pd = 10.0e-12 #F    Photodiode capacitance
I_pd = 1.5e-12  #A/sqrt(Hz)     Currentnoise
P_tel = 2.0     #W   Laser power transmitted through the telescope
eps_opt = 0.853 #Totalopticalefficiency
#L = 3e9         #m
D = 0.4         #cm     Diameter of telescope

from scipy.special import jv
m = 0   #modulation depth

#lamb = 1064.0e-9    #激光波长
   
#f_uso = 0   #ultrastable oscillator frequency
def Sx(paras,L,lamb):
   
    f_low,f_up,f_adc,f_pt,f_uso,m,P_loc = paras     #一共七个优化参数

    

    f_het = f_up    #beat-note frequency
    j0 = jv(0,m)    #贝塞尔函数,下同
    j1 = jv(1,m)
   
    P_rec = 0.4073*pi**2*D**4*P_tel*eps_opt/(8*L**2*lamb**2)     #吸收功率

    #相位噪声
    phi_sn = np.sqrt(2*e*(P_loc+P_rec)/(R_pd*eps_het*P_loc*P_rec))
    phi_rin = RIN*np.sqrt((P_loc**2+P_rec**2)/(2*eps_het*P_loc*P_rec))
    phi_en = (np.sqrt(2*N_pd)/R_pd)*np.sqrt((I_pd**2+(2*pi*C_pd*f_up*U_pd)**2)/(eps_het*P_loc*P_rec))
    phi_tot = np.sqrt(phi_sn**2+phi_rin**2+phi_en**2)

    #carrier noise and sideband noise
    Sx_car = (lamb/2/pi)*(1/j0**2)*phi_tot
    Sx_sb = (1/np.sqrt(2))*(lamb/2/pi)*(f_het/f_uso)*(1/j1**2)*phi_tot
    
    return Sx_car,Sx_sb

# para_test = [5.0e6,25.0e6,59.0e6,57.0e6,3.363e9,0.44,2.06e-3]
# sx_result = Sx(para_test,L=3.0e9,lamb=1064.0e-9)[0]+Sx(para_test,L=3.0e9,lamb=1064.0e-9)[1]
# print(f"总 Sa: {sx_result:.3e} meters")
#根据robson的灵敏度曲线，修改为带有测距噪声的版本
def PSD_Sx(f,paras):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    L,lamb=paras[-2:]
    paras_for_Sx=paras[:7]
    #L = 3.0*10**9   # Length of LISA arm
    f0 = c/(2*pi*L)   
    lamb=lamb*1.0e-9
    #这里将默认的参数替换成优化后的参数
    sx = Sx(paras_for_Sx,L,lamb)
    sx = sx[0]+sx[1]
    sa = 3*10**-15
    Poms = (sx**2)*(coeff[0]*L*L+coeff[1])*(1 + ((2*10**-3)/f)**4)*lamb/1064.0e-9/(8.0*10**-12 ) # Optical Metrology Sensor
    Pacc = sa**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)*(1064.0e-9/lamb)**2 # Acceleration Noise


    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**alpha + beta * f * np.sin(k * f)) * (1 \
                                        + np.tanh(gamma*(f_k - f)))   # Confusion noise
    
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD


def check_constraints(
    delta_x_sideband,  # δx_sideband 来自Sx[1]
    delta_x_total,     # δx^total 来自Sx[0]+Sx[1]
    f_adc,     # 单位 Hz
    f_pt,      # 单位 Hz
    f_low,   # 单位 Hz
    f_up,   # 单位 Hz
    f_uso,     # 单位 Hz
    m
):
    """
    根据所给约束，判断是否全部满足。
    单位约定：
      - 各频率 (f_ADC, f_PT, f_lower, f_upper, f_USO) 使用 Hz
      - δx_sideband, δx_total 使用同样的位移单位
      - m 为无量纲

    返回值:
      True  - 若所有约束均满足
      False - 若有任一约束不满足
    """

    # 1) δx_sideband < (1/10)·δx_total
    if not (delta_x_sideband < 0.1 * delta_x_total):
        return False

    # 2) 判断f_USO能否被f_ADC整除
    if not (f_uso % f_adc == 0):
        return False

    # 3) 判断f_USO能否被f_PT整除
    if not (f_uso % f_pt == 0):
        return False

    # 4) f_ADC > 2 * f_upper
    if not (f_adc > 2 * f_up):
        return False

    # 5) |f_ADC - f_PT| < f_lower
    if not (abs(f_adc - f_pt) < f_low):
        return False

    # 6) f_lower >= 2 MHz  (此处 f_lower 已是 Hz, 因此 2 MHz = 2e6 Hz)
    if not (f_low >= 2e6):
        return False

    # 7) f_upper <= 25 MHz (同理，25 MHz = 25e6 Hz)
    if not (f_up <= 25e6):
        return False

    # 8) f_USO < 5 GHz     (5 GHz = 5e9 Hz)
    if not (f_uso < 5e9):
        return False

    # 9) f_PT < 98 MHz    (98 MHz = 98e6 Hz)
    if not (f_pt < 98e6):
        return False

    # 10) 0.44 < m < 0.61
    if not (0.44 < m < 0.61):
        return False

    # 如果所有判断都通过，则返回 True
    return True
