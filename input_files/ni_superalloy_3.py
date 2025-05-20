import numpy as np
import taichi as ti
 

# solutes     = ['AL', 'CO'] #溶質
# uini_ave    = [0.19, 0.05] # pycalphad計算用
# yini_FCC_A1 = np.array([[0.1551411882441475, 0.06823364197490378]])
# yini_L12    = np.array([[0.8855250854215138, 0.05673927292507778],
#                         [0.0015594619850827906, 0.02505167851088545]])


solutes     = ['AL', 'CO']
uini_ave    = np.array([0.189, 0.05])
yini_FCC_A1 = np.array([[0.15506883819959466, 0.06752548056127561]])
yini_L12 = np.array([[0.8856662996928424, 0.056340461040633255],
                   [0.001553108484405842, 0.02476007102309915]])



# -----------------------------------------------CALPHAD関数-------------------------------------------------------


# @ti.func
# def f_FCC_A1(T:float,AL1:float,CO1:float) -> float:
#     NI1 = (1.0-AL1-CO1)
#     return -500.0*AL1*CO1*(-AL1 - CO1 + 1.0) - 105000.0*AL1*CO1 - 131234.176*AL1*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**3 + 78794.86352*AL1*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 + 24537.9628*AL1*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) - 139709.599*AL1*(-AL1 - CO1 + 1.0) + 11640.3*AL1*ti.log(AL1) + 4658.42657161987*AL1 + 968.06*CO1*(-AL1 - CO1 + 1.0) + 11640.3*CO1*ti.log(CO1) + 307.3643844036396*CO1 + 11640.3*(-AL1 - CO1 + 1.0)*ti.log(-AL1 - CO1 + 1.0) - 73767.16265868029

# @ti.func
# def fyy_FCC_A1_AL1(T:float,AL1:float,CO1:float) -> float:
#     fyy_AL1 = -393702.528*AL1*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) + 157589.72704*AL1*(-AL1 - CO1 + 1.0) + 787405.056*AL1*(AL1 + CO1/2 - 0.5)**2 - 157589.72704*AL1*(2*AL1 + CO1 - 1.0) - 294455.5536*AL1 - 146227.7768*CO1 - 3*(-131234.176*AL1 - 131234.176*CO1 + 131234.176)*(AL1 + CO1/2 - 0.5)**2 + (-78794.86352*AL1 - 78794.86352*CO1 + 78794.86352)*(2*AL1 + CO1 - 1.0) - 393702.528*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 + 78794.86352*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) + 262468.352*(AL1 + CO1/2 - 0.5)**3 - 157589.72704*(AL1 + CO1/2 - 0.5)**2 + 426646.9748 + 11640.3/(-AL1 - CO1 + 1.0) + 11640.3/AL1
#     return fyy_AL1
# @ti.func
# def fyy_FCC_A1_CO1(T:float,AL1:float,CO1:float) -> float:
#     fyy_CO1 = -196851.264*AL1*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5) + 39397.43176*AL1*(-AL1 - CO1 + 1.0) + 393702.528*AL1*(AL1 + CO1/2 - 0.5)**2 - 157589.72704*AL1*(AL1 + CO1/2 - 0.5) - 48075.9256*AL1 - 1936.12 + 11640.3/(-AL1 - CO1 + 1.0) + 11640.3/CO1
#     return fyy_CO1

# @ti.func
# def f_L12(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
#     NI1 = (1.0-AL1-CO1)
#     NI2 = (1.0-AL2-CO2)
#     return -79000.0*AL1*AL2*(-AL2 - CO2 + 1.0) - 64108.73608706042*AL1*AL2 + 1.0e-8*AL1*CO1*(AL1 - CO1)**2*(-AL2 - CO2 + 1.0) + 2500.0*AL1*CO1*(AL1 - CO1)*(-AL2 - CO2 + 1.0) + 800.0*AL1*CO1*(-AL2 - CO2 + 1.0) - 93046.71347747259*AL1*CO2 - 800.0*AL1*(-AL1 - CO1 + 1.0)*(-AL2 - CO2 + 1.0) - 107902.5560157753*AL1*(-AL2 - CO2 + 1.0) + 2910.075*AL1*ti.log(AL1) - 92688.06188386448*AL2*CO1 - 70273.34272996539*AL2*(-AL1 - CO1 + 1.0) + 8730.225*AL2*ti.log(AL2) - 58026.03927427664*CO1*CO2 - 79381.88181257938*CO1*(-AL2 - CO2 + 1.0) + 2910.075*CO1*ti.log(CO1) - 73211.32012037755*CO2*(-AL1 - CO1 + 1.0) + 8730.225*CO2*ti.log(CO2) - 69867.16265868029*(-AL1 - CO1 + 1.0)*(-AL2 - CO2 + 1.0) + 2910.075*(-AL1 - CO1 + 1.0)*ti.log(-AL1 - CO1 + 1.0) + 8730.225*(-AL2 - CO2 + 1.0)*ti.log(-AL2 - CO2 + 1.0)

# @ti.func
# def fyy_L12_AL1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
#     fyy_AL1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 + 1.0) - 1600.0*AL2 + 2.0e-8*CO1*(2*AL1 - 2*CO1)*(-AL2 - CO2 + 1.0) + 5000.0*CO1*(-AL2 - CO2 + 1.0) - 1600.0*CO2 + 1600.0 + 2910.075/(-AL1 - CO1 + 1.0) + 2910.075/AL1
#     return fyy_AL1
# @ti.func
# def fyy_L12_CO1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
#     fyy_CO1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 + 1.0) + 2.0e-8*AL1*(-2*AL1 + 2*CO1)*(-AL2 - CO2 + 1.0) - 5000.0*AL1*(-AL2 - CO2 + 1.0) + 2910.075/(-AL1 - CO1 + 1.0) + 2910.075/CO1
#     return fyy_CO1
# @ti.func
# def fyy_L12_AL2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
#     fyy_AL2 = 158000.0*AL1 + 8730.225/(-AL2 - CO2 + 1.0) + 8730.225/AL2
#     return fyy_AL2
# @ti.func
# def fyy_L12_CO2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
#     fyy_CO2 = 8730.225/(-AL2 - CO2 + 1.0) + 8730.225/CO2
#     return fyy_CO2


# @ti.func
# def cal_f(i:int,T:float,y:float) -> float:
#     f = 0.0
#     if   i == 0:
#         f = f_FCC_A1(T,y[0,0],y[0,1])
#     elif i == 1  or i == 2 or i == 3 or i == 4:
#         A = i*2 - 1
#         f = f_L12(T,y[A,0],y[A,1],y[A+1,0],y[A+1,1])
#     return f

# @ti.func
# def cal_fyy(a:int,j:int,T:float,y:float) -> float:
#     fyy = 0.0
#     if   a == 0: # if sublattice 1 in FCC_A1
#         if   j == 0: fyy = fyy_FCC_A1_AL1(T,y[0,0],y[0,1])
#         elif j == 1: fyy = fyy_FCC_A1_CO1(T,y[0,0],y[0,1])
#     elif a == 1 or a == 3 or a == 5 or a == 7: # if sublattice 1 in L12
#         if   j == 0: fyy = fyy_L12_AL1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
#         elif j == 1: fyy = fyy_L12_CO1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
#     elif a == 2 or a == 4 or a == 6 or a == 8: # if sublattice 2 in L12
#         if   j == 0: fyy = fyy_L12_AL2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
#         elif j == 1: fyy = fyy_L12_CO2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
#     return fyy





@ti.func
def GHSERAL(T):
    return -69108.736087060417
@ti.func
def GHSERCO(T):
    return -73026.039274276642
@ti.func
def GHSERCR(T):
    return -61524.757540082690
@ti.func
def GHSERFE(T):
    return -72192.393240383986
@ti.func
def GHSERHF(T):
    return -90387.164237073972
@ti.func
def GHSERMO(T):
    return -67695.442820141296
@ti.func
def GHSERNB(T):
    return -78618.604652174399
@ti.func
def GHSERNI(T):
    return -73767.162658680289
@ti.func
def GHSERSI(T):
    return -51221.635893172890
@ti.func
def GHSERTI(T):
    return -73377.570647175264
@ti.func
def GHSERWW(T):
    return -72906.070942171311
@ti.func
def GHSERZR(T):
    return -85521.656255392678
@ti.func
def GCOFCC(T):
    return -73459.798274276647
@ti.func
def GFEFCC(T):
    return -72468.799540505381
@ti.func
def GHFFCC(T):
    return -83466.916437073989
@ti.func
def GTIFCC(T):
    return -67517.570647175264
@ti.func
def GZRFCC(T):
    return -79181.656255392678
@ti.func
def GALHCP(T):
    return -66147.736087060417
@ti.func
def GFEHCP(T):
    return -68679.579814905432
@ti.func
def GNBHCP(T):
    return -65258.604652174399
@ti.func
def GNIHCP(T):
    return -70963.882658680290
@ti.func
def GHEXTNB(T):
    return -78615.091491031548
@ti.func
def GDHCFE(T):
    return -70574.189677705406
@ti.func
def GDHCNI(T):
    return -72365.522658680289
@ti.func
def GDHCTI(T):
    return -70447.570647175264
@ti.func
def GFCC_A1FEVA0(T):
    return -72468.799540505381


# [['AL']] 0 : (+GHSERAL)
# [['CO']] 0 : (+GCOFCC)
# [['NI']] 0 : (+GHSERNI)
# [['AL', 'CO']] 0 : (-105000)
# [['AL', 'NI']] 0 : (-162407.75+16.212965*T)
# [['AL', 'NI']] 1 : (+73417.798-34.914168*T)
# [['AL', 'NI']] 2 : (+33471.014-9.8373558*T)
# [['AL', 'NI']] 3 : (-30758.01+10.25267*T)
# [['CO', 'NI']] 0 : (-800+1.2629*T)
# [['AL', 'CO', 'NI']] 0 : (-7500+5*T)
# ternary omit=True
@ti.func
def f_FCC_A1(T:float,AL1:float,CO1:float) -> float:
    NI1 = (1.0-AL1-CO1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CO1)*ti.log(CO1) +(NI1)*ti.log(NI1) ) )+ \
           + AL1                                       *(+GHSERAL(T))+ \
           + CO1                                       *(+GCOFCC(T))+ \
           + NI1                                       *(+GHSERNI(T))+ \
           + AL1*CO1                                   *(-105000)+ \
           + AL1*NI1                                   *(-162407.75+16.212965*T)+ \
           + AL1*NI1*(AL1-NI1)                         *(+73417.798-34.914168*T)+ \
           + AL1*NI1*(AL1-NI1)**2                      *(+33471.014-9.8373558*T)+ \
           + AL1*NI1*(AL1-NI1)**3                      *(-30758.01+10.25267*T)+ \
           + CO1*NI1                                   *(-800+1.2629*T)+ \
           + AL1*CO1*NI1                               *(-7500+5*T) )

@ti.func
def fy_FCC_A1_AL1(T:float,AL1:float,CO1:float) -> float:
    fy_AL1 = -1.0*AL1*CO1*(5*T - 7500) + 4.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) - 4.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 - 0.5)**2 + 2.0*AL1*(73417.798 - 34.914168*T)*(-AL1 - CO1 + 1.0) - 1.0*AL1*(73417.798 - 34.914168*T)*(2*AL1 + CO1 - 1.0) + 24.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 - 8.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 - 0.5)**3 - 1.0*AL1*(16.212965*T - 162407.75) - 1.0*CO1*(1.2629*T - 800) + 1.0*CO1*(5*T - 7500)*(-AL1 - CO1 + 1.0) - 105000.0*CO1 + 8.3145*T*(1.0*ti.log(AL1) - 1.0*ti.log(-AL1 - CO1 + 1.0)) + 4.0*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 + 1.0*(73417.798 - 34.914168*T)*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) + 8.0*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**3 + 1.0*(16.212965*T - 162407.75)*(-AL1 - CO1 + 1.0) + 1.0*GHSERAL(T) - 1.0*GHSERNI(T)
    return fy_AL1
@ti.func
def fy_FCC_A1_CO1(T:float,AL1:float,CO1:float) -> float:
    fy_CO1 = -1.0*AL1*CO1*(5*T - 7500) + 4.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5) - 4.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 - 0.5)**2 + 1.0*AL1*(73417.798 - 34.914168*T)*(-AL1 - CO1 + 1.0) - 1.0*AL1*(73417.798 - 34.914168*T)*(2*AL1 + CO1 - 1.0) + 1.0*AL1*(5*T - 7500)*(-AL1 - CO1 + 1.0) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 - 8.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 - 0.5)**3 - 1.0*AL1*(16.212965*T - 162407.75) - 105000.0*AL1 - 1.0*CO1*(1.2629*T - 800) + 8.3145*T*(1.0*ti.log(CO1) - 1.0*ti.log(-AL1 - CO1 + 1.0)) + 1.0*(1.2629*T - 800)*(-AL1 - CO1 + 1.0) + 1.0*GCOFCC(T) - 1.0*GHSERNI(T)
    return fy_CO1

@ti.func
def fyy_FCC_A1_AL1(T:float,AL1:float,CO1:float) -> float:
    fyy_AL1 = 8.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(2*AL1 + CO1 - 1.0) - 4.0*AL1*(73417.798 - 34.914168*T) + 24.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) - 48.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 - 0.5)**2 - 2.0*CO1*(5*T - 7500) + 8.3145*T*(1.0/(-AL1 - CO1 + 1.0) + 1.0/AL1) - 32.42593*T + 4.0*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) - 4.0*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 - 0.5)**2 + 4.0*(73417.798 - 34.914168*T)*(-AL1 - CO1 + 1.0) - 2.0*(73417.798 - 34.914168*T)*(2*AL1 + CO1 - 1.0) + (133884.056 - 39.3494232*T)*(-AL1 - CO1 + 1.0)*(2*AL1 + CO1 - 1.0) - (133884.056 - 39.3494232*T)*(AL1 + CO1/2 - 0.5)**2 + 24.0*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 - 8.0*(10.25267*T - 30758.01)*(AL1 + CO1/2 - 0.5)**3 + 3*(82.02136*T - 246064.08)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5)**2 - (82.02136*T - 246064.08)*(AL1 + CO1/2 - 0.5)**3 + 324815.5
    return fyy_AL1
@ti.func
def fyy_FCC_A1_CO1(T:float,AL1:float,CO1:float) -> float:
    fyy_CO1 = 2.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 - 0.5) - 2.0*AL1*(73417.798 - 34.914168*T) - 2.0*AL1*(5*T - 7500) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 + 1.0)*(AL1 + CO1/2 - 0.5) - 24.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 - 0.5)**2 + 8.3145*T*(1.0/(-AL1 - CO1 + 1.0) + 1.0/CO1) - 2.5258*T + 1600.0
    return fyy_CO1

# diployed wild card in [['*'], ['NI', 'W']]
# diployed wild card in [['*'], ['CR', 'NI']]
# diployed wild card in [['*'], ['CO', 'MO']]
# [['AL'], ['AL']] 0 : (+GHSERAL+5000)
# [['AL'], ['CO']] 0 : (-21000+0.25*GHSERAL+0.75*GHSERCO)
# [['AL'], ['NI']] 0 : (-43000+5.5*T+0.75*GHSERNI+0.25*GHSERAL)
# [['CO'], ['AL']] 0 : (+0.75*GHSERAL+0.25*GHSERCO-22600)
# [['CO'], ['CO']] 0 : (+GHSERCO+15000)
# [['CO'], ['NI']] 0 : (-1600-3*T+0.75*GHSERNI+0.25*GHSERCO)
# [['NI'], ['AL']] 0 : (+0.75*GHSERAL+0.25*GHSERNI)
# [['NI'], ['CO']] 0 : (+0.75*GHSERCO+0.25*GHSERNI)
# [['NI'], ['NI']] 0 : (+GHSERNI+3900)
# [['AL'], ['AL', 'NI']] 0 : (-100000+15*T)
# [['AL', 'NI'], ['NI']] 0 : (+2000-2*T)
# [['AL', 'CO'], ['NI']] 0 : (+5000-3*T)
# [['AL', 'CO'], ['NI']] 1 : (+2500)
# [['AL', 'CO'], ['NI']] 2 : (+1E-8)
@ti.func
def f_L12(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    NI1 = (1.0-AL1-CO1)
    NI2 = (1.0-AL2-CO2)
    return 1/1.0*(+ \
           + 8.3145*T*( +0.25*(+(AL1)*ti.log(AL1) +(CO1)*ti.log(CO1) +(NI1)*ti.log(NI1) )  +0.75*(+(AL2)*ti.log(AL2) +(CO2)*ti.log(CO2) +(NI2)*ti.log(NI2) ) )+ \
           + AL1                                       *AL2                                       *(+GHSERAL(T)+5000)+ \
           + AL1                                       *CO2                                       *(-21000+0.25*GHSERAL(T)+0.75*GHSERCO(T))+ \
           + AL1                                       *NI2                                       *(-43000+5.5*T+0.75*GHSERNI(T)+0.25*GHSERAL(T))+ \
           + CO1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERCO(T)-22600)+ \
           + CO1                                       *CO2                                       *(+GHSERCO(T)+15000)+ \
           + CO1                                       *NI2                                       *(-1600-3*T+0.75*GHSERNI(T)+0.25*GHSERCO(T))+ \
           + NI1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERNI(T))+ \
           + NI1                                       *CO2                                       *(+0.75*GHSERCO(T)+0.25*GHSERNI(T))+ \
           + NI1                                       *NI2                                       *(+GHSERNI(T)+3900)+ \
           + AL1                                       *AL2*NI2                                   *(-100000+15*T)+ \
           + AL1*NI1                                   *NI2                                       *(+2000-2*T)+ \
           + AL1*CO1                                   *NI2                                       *(+5000-3*T)+ \
           + AL1*CO1*(AL1-CO1)                         *NI2                                       *(+2500)+ \
           + AL1*CO1*(AL1-CO1)**2                      *NI2                                       *(+1E-8) )

@ti.func
def fy_L12_AL1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fy_AL1 = 1.0e-8*AL1*CO1*(2*AL1 - 2*CO1)*(-AL2 - CO2 + 1.0) + 2500.0*AL1*CO1*(-AL2 - CO2 + 1.0) - 1.0*AL1*(2000 - 2*T)*(-AL2 - CO2 + 1.0) + 1.0*AL2*(15*T - 100000)*(-AL2 - CO2 + 1.0) - 1.0*AL2*(0.75*GHSERAL(T) + 0.25*GHSERNI(T)) + 1.0*AL2*(GHSERAL(T) + 5000) + 1.0*CO1*(5000 - 3*T)*(-AL2 - CO2 + 1.0) + 1.0e-8*CO1*(AL1 - CO1)**2*(-AL2 - CO2 + 1.0) + 2500.0*CO1*(AL1 - CO1)*(-AL2 - CO2 + 1.0) - 1.0*CO2*(0.75*GHSERCO(T) + 0.25*GHSERNI(T)) + 1.0*CO2*(0.25*GHSERAL(T) + 0.75*GHSERCO(T) - 21000) + 8.3145*T*(0.25*ti.log(AL1) - 0.25*ti.log(-AL1 - CO1 + 1.0)) + 1.0*(2000 - 2*T)*(-AL1 - CO1 + 1.0)*(-AL2 - CO2 + 1.0) - 1.0*(GHSERNI(T) + 3900)*(-AL2 - CO2 + 1.0) + 1.0*(-AL2 - CO2 + 1.0)*(5.5*T + 0.25*GHSERAL(T) + 0.75*GHSERNI(T) - 43000)
    return fy_AL1
@ti.func
def fy_L12_CO1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fy_CO1 = 1.0e-8*AL1*CO1*(-2*AL1 + 2*CO1)*(-AL2 - CO2 + 1.0) - 2500.0*AL1*CO1*(-AL2 - CO2 + 1.0) - 1.0*AL1*(2000 - 2*T)*(-AL2 - CO2 + 1.0) + 1.0*AL1*(5000 - 3*T)*(-AL2 - CO2 + 1.0) + 1.0e-8*AL1*(AL1 - CO1)**2*(-AL2 - CO2 + 1.0) + 2500.0*AL1*(AL1 - CO1)*(-AL2 - CO2 + 1.0) - 1.0*AL2*(0.75*GHSERAL(T) + 0.25*GHSERNI(T)) + 1.0*AL2*(0.75*GHSERAL(T) + 0.25*GHSERCO(T) - 22600) - 1.0*CO2*(0.75*GHSERCO(T) + 0.25*GHSERNI(T)) + 1.0*CO2*(GHSERCO(T) + 15000) + 8.3145*T*(0.25*ti.log(CO1) - 0.25*ti.log(-AL1 - CO1 + 1.0)) - 1.0*(GHSERNI(T) + 3900)*(-AL2 - CO2 + 1.0) + 1.0*(-AL2 - CO2 + 1.0)*(-3*T + 0.25*GHSERCO(T) + 0.75*GHSERNI(T) - 1600)
    return fy_CO1
@ti.func
def fy_L12_AL2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fy_AL2 = -1.0*AL1*AL2*(15*T - 100000) - 1.0*AL1*CO1*(5000 - 3*T) - 1.0e-8*AL1*CO1*(AL1 - CO1)**2 - 2500.0*AL1*CO1*(AL1 - CO1) - 1.0*AL1*(2000 - 2*T)*(-AL1 - CO1 + 1.0) + 1.0*AL1*(15*T - 100000)*(-AL2 - CO2 + 1.0) + 1.0*AL1*(GHSERAL(T) + 5000) - 1.0*AL1*(5.5*T + 0.25*GHSERAL(T) + 0.75*GHSERNI(T) - 43000) + 1.0*CO1*(0.75*GHSERAL(T) + 0.25*GHSERCO(T) - 22600) - 1.0*CO1*(-3*T + 0.25*GHSERCO(T) + 0.75*GHSERNI(T) - 1600) + 8.3145*T*(0.75*ti.log(AL2) - 0.75*ti.log(-AL2 - CO2 + 1.0)) + 1.0*(0.75*GHSERAL(T) + 0.25*GHSERNI(T))*(-AL1 - CO1 + 1.0) - 1.0*(GHSERNI(T) + 3900)*(-AL1 - CO1 + 1.0)
    return fy_AL2
@ti.func
def fy_L12_CO2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fy_CO2 = -1.0*AL1*AL2*(15*T - 100000) - 1.0*AL1*CO1*(5000 - 3*T) - 1.0e-8*AL1*CO1*(AL1 - CO1)**2 - 2500.0*AL1*CO1*(AL1 - CO1) - 1.0*AL1*(2000 - 2*T)*(-AL1 - CO1 + 1.0) + 1.0*AL1*(0.25*GHSERAL(T) + 0.75*GHSERCO(T) - 21000) - 1.0*AL1*(5.5*T + 0.25*GHSERAL(T) + 0.75*GHSERNI(T) - 43000) + 1.0*CO1*(GHSERCO(T) + 15000) - 1.0*CO1*(-3*T + 0.25*GHSERCO(T) + 0.75*GHSERNI(T) - 1600) + 8.3145*T*(0.75*ti.log(CO2) - 0.75*ti.log(-AL2 - CO2 + 1.0)) + 1.0*(0.75*GHSERCO(T) + 0.25*GHSERNI(T))*(-AL1 - CO1 + 1.0) - 1.0*(GHSERNI(T) + 3900)*(-AL1 - CO1 + 1.0)
    return fy_CO2

@ti.func
def fyy_L12_AL1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fyy_AL1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 + 1.0) + 2.0e-8*CO1*(2*AL1 - 2*CO1)*(-AL2 - CO2 + 1.0) + 5000.0*CO1*(-AL2 - CO2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 + 1.0) + 0.25/AL1) - 1.0*(2000 - 2*T)*(-AL2 - CO2 + 1.0) - (2000.0 - 2.0*T)*(-AL2 - CO2 + 1.0)
    return fyy_AL1
@ti.func
def fyy_L12_CO1(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fyy_CO1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 + 1.0) + 2.0e-8*AL1*(-2*AL1 + 2*CO1)*(-AL2 - CO2 + 1.0) - 5000.0*AL1*(-AL2 - CO2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 + 1.0) + 0.25/CO1)
    return fyy_CO1
@ti.func
def fyy_L12_AL2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fyy_AL2 = -2.0*AL1*(15*T - 100000) + 8.3145*T*(0.75/(-AL2 - CO2 + 1.0) + 0.75/AL2)
    return fyy_AL2
@ti.func
def fyy_L12_CO2(T:float,AL1:float,CO1:float,AL2:float,CO2:float) -> float:
    fyy_CO2 = 8.3145*T*(0.75/(-AL2 - CO2 + 1.0) + 0.75/CO2)
    return fyy_CO2


@ti.func
def cal_f(i:int,T:float,y:float) -> float:
    f = 0.0
    if   i == 0:
        f = f_FCC_A1(T,y[0,0],y[0,1])
    elif i == 1  or i == 2 or i == 3 or i == 4:
        A = i*2 - 1
        f = f_L12(T,y[A,0],y[A,1],y[A+1,0],y[A+1,1])
    return f

@ti.func
def cal_fy(a:int,j:int,T:float,y:float) -> float:
    fy = 0.0
    if   a == 0: # if sublattice 1 in FCC_A1
        if   j == 0: fy = fy_FCC_A1_AL1(T,y[0,0],y[0,1])
        elif j == 1: fy = fy_FCC_A1_CO1(T,y[0,0],y[0,1])
    elif a == 1 or a == 3 or a == 5 or a == 7: # if sublattice 1 in L12
        if   j == 0: fy = fy_L12_AL1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
        elif j == 1: fy = fy_L12_CO1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
    elif a == 2 or a == 4 or a == 6 or a == 8: # if sublattice 2 in L12
        if   j == 0: fy = fy_L12_AL2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
        elif j == 1: fy = fy_L12_CO2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
    return fy

@ti.func
def cal_fyy(a:int,j:int,T:float,y:float) -> float:
    fyy = 0.0
    if   a == 0: # if sublattice 1 in FCC_A1
        if   j == 0: fyy = fyy_FCC_A1_AL1(T,y[0,0],y[0,1])
        elif j == 1: fyy = fyy_FCC_A1_CO1(T,y[0,0],y[0,1])
    elif a == 1 or a == 3 or a == 5 or a == 7: # if sublattice 1 in L12
        if   j == 0: fyy = fyy_L12_AL1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
        elif j == 1: fyy = fyy_L12_CO1(T,y[a,0],y[a,1],y[a+1,0],y[a+1,1])
    elif a == 2 or a == 4 or a == 6 or a == 8: # if sublattice 2 in L12
        if   j == 0: fyy = fyy_L12_AL2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
        elif j == 1: fyy = fyy_L12_CO2(T,y[a-1,0],y[a-1,1],y[a,0],y[a,1])
    return fyy

