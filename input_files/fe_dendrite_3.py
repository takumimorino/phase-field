import numpy as np
import taichi as ti
 
solutes     = ['CR', 'NI']
uini_ave    = np.array([0.226, 0.1])
yini_LIQUID = np.array([[0.23067682765285058, 0.09848095004050882]])
yini_FCC_A1 = np.array([[0.20039874800431154, 0.10831537608265704]])


# solvent  : FE
# solutes  : ['CR', 'NI']
# phases   : ['LIQUID', 'FCC_A1']

@ti.func
def GHSERAL(T): return -11278.378+188.684153*T-31.748192*T*ti.log(T)-1.231E+28*(1/T)**(9)

@ti.func
def GHSERCO(T): return +310.241+133.36601*T-25.0861*T*ti.log(T)-2.6547387E-3*T**2-1.7348E-07*T**3+72526.9*(1/T)**(1)

@ti.func
def GHSERCR(T): return -8856.94+157.48*T-26.908*T*ti.log(T)+0.00189435*T**2-1.47721E-6*T**3+139250*(1/T)**(1)

@ti.func
def GHSERCU(T): return -13542.026+183.803828*T-31.38*T*ti.log(T)+3.642E+29*(1/T)**(9)

@ti.func
def GHSERFE(T): return +1225.7+124.134*T-23.5143*T*ti.log(T)-0.00439752*T**2-5.89269E-8*T**3+77358.5*(1/T)**(1)

@ti.func
def GHSERMN(T): return -28733.41+312.2648*T-48.0*T*ti.log(T)+1.656847E+30*(1/T)**(9)

@ti.func
def GHSERMO(T): return -7746.302+131.9197*T-23.56414*T*ti.log(T)-0.003443396*T**2+5.662834E-7*T**3-1.309265E-10*T**4+65812.39*(1/T)**(1)

@ti.func
def GHSERNB(T): return -8519.353+142.045475*T-26.4711*T*ti.log(T)+0.203475E-3*T**2-0.35011E-6*T**3+93399*(1/T)**(1)

@ti.func
def GHSERNI(T): return -5179.159+117.854*T-22.096*T*ti.log(T)-4.8407E-3*T**2

@ti.func
def GHSERPD(T): return +917.062+49.659892*T-13.5708*T*ti.log(T)-0.00717522*T**2+1.91115E-07*T**3-1112465*(1/T)**(1)

@ti.func
def GHSERSI(T):
    x = 0.0
    if T >= 273.00 and T < 1687.00:
        x =  -8162.609+137.236859*T-22.8317533*T*ti.log(T)-1.912904E-3*T**2-0.003552E-6*T**3+176667*(1/T)**(1)
    elif T >= 1687.00 and T < 6000.00:
        x =  -9457.642+167.281367*T-27.196*T*ti.log(T)-4.2037E+30*(1/T)**(9)
    return x

@ti.func
def GHSERSS(T):    return -6513.769+94.692922*T-17.941839*T*ti.log(T)-0.010895125*T**2+1.402558E-06*T**3+39910*(1/T)**(1)

@ti.func
def GHSERTA(T):    return -22389.955+243.88676*T-41.137088*T*ti.log(T)+0.006167572*T**2-6.55136E-07*T**3+2429586*(1/T)**(1)

@ti.func
def GHSERTI(T):    return +908.837+66.976538*T-14.9466*T*ti.log(T)-8.1465E-3*T**2+2.02715E-07*T**3-1477660*(1/T)**(1)

@ti.func
def GHSERVV(T):    return -7967.842+143.291093*T-25.9*T*ti.log(T)+0.0625E-3*T**2-0.68E-6*T**3

@ti.func
def GHSERWW(T):    return -7646.311+130.4*T-24.1*T*ti.log(T)-1.936E-3*T**2+2.07E-7*T**3-5.33E-11*T**4+44500*(1/T)**(1)

@ti.func
def GHSERYY(T):    return -15802.62+229.831717*T-40.2851*T*ti.log(T)+6.8095E-3*T**2-1.14182E-6*T**3

@ti.func
def GCOFCC(T):    return +427.591-0.61525*T+GHSERCO(T)

@ti.func
def GHFFCC(T):    return +3012.703+108.544*T-22.7075*T*ti.log(T)-0.004146*T**2-0.000477E-6*T**3-22590*(1/T)**(1)

@ti.func
def GLAFCC(T):    return -124598.976+955.878375*T-139.346741*T*ti.log(T)+0.042032405*T**2-3.066199E-06*T**3+20994153*(1/T)**(1)

@ti.func
def GPFCC(T):    return +12294.881+140.701181*T-26.326*T*ti.log(T)

@ti.func
def GTAFCC(T):    return +16000+1.7*T+GHSERTA(T)

@ti.func
def GHEXTNB(T):    return -8519.35+142.048*T-26.4711*T*ti.log(T)+0.203475E-3*T**2-0.350119E-6*T**3+93398.8*(1/T)**(1)

@ti.func
def GSSLIQ(T):    return -6889.972+176.37082*T-32*T*ti.log(T)

@ti.func
def LF0(T):    return -20455.7462-11.7930695*T

@ti.func
def LF1(T):    return +15581.5125-8.49116947*T

@ti.func
def UNDEF(T):    return +1E-8

@ti.func
def GLIQUIDAL0(T):    return +10482.382-11.253974*T+1.231E+28*(1/T)**(9)+GHSERAL(T)

@ti.func
def GLIQUIDCO0(T):    return +15085.037-8.931932*T-2.19801E-21*T**(7)+GHSERCO(T)

@ti.func
def GLIQUIDCR0(T):    return +24339.955-11.420225*T+2.37615E-21*T**7+GHSERCR(T)

@ti.func
def GLIQUIDCU0(T):    return +13495.481-9.922344*T-3.642E+29*(1/T)**(9)+GHSERCU(T)

@ti.func
def GLIQUIDFE0(T):    return +12040.17-6.55843*T-3.6751551E-21*T**7+GHSERFE(T)

@ti.func
def GLIQUIDLA0(T):    return -3942.004+171.018431*T-34.3088*T*ti.log(T)

@ti.func
def GLIQUIDHF0(T):    return +49731.499-149.91739*T+12.116812*T*ti.log(T)-21.262021E-3*T**2+1.376466E-6*T**3-4449699*(1/T)**(1)

@ti.func
def GLIQUIDMN0(T):    return +18739.51-13.2288*T-1.656847E+30*(1/T)**(9)+GHSERMN(T)

@ti.func
def GLIQUIDMO0(T):    return +41831.347-14.694912*T+4.24519E-22*T**7+GHSERMO(T)

@ti.func
def GLIQUIDNB0(T):    return +29781.555-10.816417*T-3.06098E-23*T**7+GHSERNB(T)

@ti.func
def GLIQUIDNI0(T):    return +16414.686-9.397*T-3.82318E-21*T**7+GHSERNI(T)

@ti.func
def GLIQUIDP0(T):    return -7232.449+135.291873*T-26.326*T*ti.log(T)

@ti.func
def GLIQUIDPD0(T):    return +23405.778-116.918419*T+10.8922031*T*ti.log(T)-27.266568E-03*T**2+2.430675E-06*T**3-1853674*(1/T)**(1)

@ti.func
def GLIQUIDSI0(T):
    x = 0.0
    if T >= 273.00 and T < 1687.00:
        x =  +50696.36-30.099439*T+2.0931E-21*T**7+GHSERSI(T)
    elif T >= 1687.00 and T < 6000.00:
        x =  +49828.165-29.559068*T+4.2037E+30*(1/T)**(9)+GHSERSI(T)
    return x

@ti.func
def GLIQUIDTA0(T):    return +66274.294-305.868555*T+41.1650403*T*ti.log(T)-0.018497638*T**2+1.269735E-06*T**3-5952924*(1/T)**(1)+GHSERTA(T)

@ti.func
def GLIQUIDTI0(T):    return +368610.36-2620.9995038*T+357.005867*T*ti.log(T)-155.262855E-3*T**2+12.254402E-6*T**3-65556856*(1/T)**(1)+GHSERTI(T)

@ti.func
def GLIQUIDV0(T):    return +20764.117-9.455552*T-5.19136E-22*T**7+GHSERVV(T)

@ti.func
def GLIQUIDW0(T):    return +52160.584-14.10999*T-2.713468E-24*T**7+GHSERWW(T)

@ti.func
def GLIQUIDY0(T):    return +3934.121+59.921688*T-14.8146562*T*ti.log(T)-15.623487E-3*T**2+1.442946E-6*T**3-140695*(1/T)**(1)

@ti.func
def GFCC_A1FEVA0(T):    return -1462.4+8.282*T-1.15*T*ti.log(T)+0.00064*T**2+GHSERFE(T)

@ti.func
def GFCC_A1MNVA0(T):    return -26070.1+309.6664*T-48*T*ti.log(T)+3.86196E+30*(1/T)**(9)

# import taichi as ti
# @ti.func
# def GHSERCR(T):
#     x = 0.0
#     if T >= 273.00 and T < 2180.00:
#         x =  -8856.94+157.48*T-26.908*T*ti.log(T)+0.00189435*T**2-1.47721E-6*T**3+139250*(1/T)**(1)
#     elif T >= 2180.00 and T < 6000.00:
#         x =  -34869.344+344.18*T-50*T*ti.log(T)-2.88526E+32*(1/T)**(9)
#     return x

# @ti.func
# def GHSERFE(T):
#     x = 0.0
#     if T >= 273.00 and T < 1811.00:
#         x =  +1225.7+124.134*T-23.5143*T*ti.log(T)-0.00439752*T**2-5.89269E-8*T**3+77358.5*(1/T)**(1)
#     elif T >= 1811.00 and T < 6000.00:
#         x =  -25383.581+299.31255*T-46*T*ti.log(T)+2.2960305E+31*(1/T)**(9)
#     return x

# @ti.func
# def GHSERNI(T):
#     x = 0.0
#     if T >= 273.00 and T < 1728.00:
#         x =  -5179.159+117.854*T-22.096*T*ti.log(T)-4.8407E-3*T**2
#     elif T >= 1728.00 and T < 6000.00:
#         x =  -27840.655+279.135*T-43.10*T*ti.log(T)+1.12754E+31*(1/T)**(9)
#     return x

# @ti.func
# def GLIQUIDCR0(T):
#     x = 0.0
#     if T >= 273.00 and T < 2180.00:
#         x =  +24339.955-11.420225*T+2.37615E-21*T**7+GHSERCR(T)
#     elif T >= 2180.00 and T < 6000.00:
#         x =  +18409.36-8.563683*T+2.88526E+32*(1/T)**(9)+GHSERCR(T)
#     return x

# @ti.func
# def GLIQUIDFE0(T):
#     x = 0.0
#     if T >= 273.00 and T < 1811.00:
#         x =  +12040.17-6.55843*T-3.6751551E-21*T**7+GHSERFE(T)
#     elif T >= 1811.00 and T < 6000.00:
#         x =  -10839.7+291.302*T-46*T*ti.log(T)
#     return x

# @ti.func
# def GLIQUIDNI0(T):
#     x = 0.0
#     if T >= 273.00 and T < 1728.00:
#         x =  +16414.686-9.397*T-3.82318E-21*T**7+GHSERNI(T)
#     elif T >= 1728.00 and T < 6000.00:
#         x =  +18290.88-10.537*T-1.12754E+31*(1/T)**(9)+GHSERNI(T)
#     return x

# @ti.func
# def GFCC_A1FEVA0(T):
#     x = 0.0
#     if T >= 273.00 and T < 1811.00:
#         x =  -1462.4+8.282*T-1.15*T*ti.log(T)+0.00064*T**2+GHSERFE(T)
#     elif T >= 1811.00 and T < 6000.00:
#         x =  -1713.815+0.94001*T+0.4925095E+31*(1/T)**(9)+GHSERFE(T)
#     return x

@ti.func
def f_LIQUID(T:float,CR1:float,NI1:float) -> float:
    FE1 = (1.0-CR1-NI1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(CR1)*ti.log(CR1) +(FE1)*ti.log(FE1) +(NI1)*ti.log(NI1) ) )+ \
           + CR1                                       *GLIQUIDCR0(T)+ \
           + FE1                                       *GLIQUIDFE0(T)+ \
           + NI1                                       *GLIQUIDNI0(T)+ \
           + CR1*FE1                                   *(-17737+7.996546*T)+ \
           + CR1*FE1*(CR1-FE1)                         *(-1331)+ \
           + CR1*NI1                                   *(+318-7.3318*T)+ \
           + CR1*NI1*(CR1-NI1)                         *(+16941-6.3696*T)+ \
           + FE1*NI1                                   *(-18378.86+6.03912*T)+ \
           + FE1*NI1*(FE1-NI1)                         *(+9228.1-3.54642*T)+ \
           + CR1*FE1*NI1*(CR1+(1/3)*(1-(CR1+FE1+NI1))) *(+140000-50*T)+ \
           + CR1*FE1*NI1*(FE1+(1/3)*(1-(CR1+FE1+NI1))) *(+5000-12*T)+ \
           + CR1*FE1*NI1*(NI1+(1/3)*(1-(CR1+FE1+NI1))) *(+75000) )

@ti.func
def fyy_LIQUID_CR1(T:float,CR1:float,NI1:float) -> float:
    fyy_CR1 = 2.0*CR1*NI1*(5000 - 12*T) - 4.0*CR1*NI1*(140000 - 50*T) + 15972.0*CR1 - 150000.0*NI1**2 + 2.0*NI1*(5000 - 12*T)*(2*CR1 + 2*NI1 - 2.0) + 2.0*NI1*(9228.1 - 3.54642*T) + 2.0*NI1*(16941 - 6.3696*T) + 2.0*NI1*(140000 - 50*T)*(-CR1 - NI1 + 1.0) + 7986.0*NI1 + 8.3145*T*(1.0/(-CR1 - NI1 + 1.0) + 1.0/CR1) - 15.993092*T + 27488.0
    return fyy_CR1
@ti.func
def fyy_LIQUID_NI1(T:float,CR1:float,NI1:float) -> float:
    fyy_NI1 = -2.0*CR1**2*(140000 - 50*T) + 2.0*CR1*NI1*(5000 - 12*T) - 300000.0*CR1*NI1 + 2.0*CR1*(5000 - 12*T)*(2*CR1 + 2*NI1 - 2.0) - 2.0*CR1*(16941 - 6.3696*T) + 150000.0*CR1*(-CR1 - NI1 + 1.0) + 2662.0*CR1 + 4.0*NI1*(9228.1 - 3.54642*T) + 8.3145*T*(1.0/(-CR1 - NI1 + 1.0) + 1.0/NI1) - 12.07824*T - 2.0*(9228.1 - 3.54642*T)*(-CR1 - 2*NI1 + 1.0) - 4.0*(9228.1 - 3.54642*T)*(-CR1 - NI1 + 1.0) + 36757.72
    return fyy_NI1

@ti.func
def f_FCC_A1(T:float,CR1:float,NI1:float) -> float:
    FE1 = (1.0-CR1-NI1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(CR1)*ti.log(CR1) +(FE1)*ti.log(FE1) +(NI1)*ti.log(NI1) ) )+ \
           + CR1                                       *(+7284+0.163*T+GHSERCR(T))+ \
           + FE1                                       *GFCC_A1FEVA0(T)+ \
           + NI1                                       *(+GHSERNI(T))+ \
           + CR1*FE1                                   *(+10833-7.477*T)+ \
           + CR1*FE1*(CR1-FE1)                         *(+1410)+ \
           + CR1*NI1                                   *(+8030-12.8801*T)+ \
           + CR1*NI1*(CR1-NI1)                         *(+33080-16.0362*T)+ \
           + FE1*NI1                                   *(-12054.355+3.27413*T)+ \
           + FE1*NI1*(FE1-NI1)                         *(+11082.1315-4.45077*T)+ \
           + FE1*NI1*(FE1-NI1)**2                      *(-725.805174)+ \
           + CR1*FE1*NI1*(CR1+(1/3)*(1-(CR1+FE1+NI1))) *(+8000-8*T)+ \
           + CR1*FE1*NI1*(FE1+(1/3)*(1-(CR1+FE1+NI1))) *(-6500)+ \
           + CR1*FE1*NI1*(NI1+(1/3)*(1-(CR1+FE1+NI1))) *(+30000) )

@ti.func
def fyy_FCC_A1_CR1(T:float,CR1:float,NI1:float) -> float:
    fyy_CR1 = -4.0*CR1*NI1*(8000 - 8*T) - 13000.0*CR1*NI1 - 16920.0*CR1 - 60000.0*NI1**2 + 2.0*NI1*(8000 - 8*T)*(-CR1 - NI1 + 1.0) + 2.0*NI1*(11082.1315 - 4.45077*T) + 2.0*NI1*(33080 - 16.0362*T) - 1451.610348*NI1*(-CR1 - NI1 + 1.0) + 5806.441392*NI1*(CR1/2 + NI1 - 0.5) - 13000.0*NI1*(2*CR1 + 2*NI1 - 2.0) - 8460.0*NI1 + 8.3145*T*(1.0/(-CR1 - NI1 + 1.0) + 1.0/CR1) + 14.954*T - 13206.0
    return fyy_CR1
@ti.func
def fyy_FCC_A1_NI1(T:float,CR1:float,NI1:float) -> float:
    fyy_NI1 = -2.0*CR1**2*(8000 - 8*T) - 133000.0*CR1*NI1 - 2.0*CR1*(33080 - 16.0362*T) + 60000.0*CR1*(-CR1 - NI1 + 1.0) - 13000.0*CR1*(2*CR1 + 2*NI1 - 2.0) - 2820.0*CR1 + 4.0*NI1*(11082.1315 - 4.45077*T) - 5806.441392*NI1*(-CR1 - NI1 + 1.0) + 5806.441392*NI1*(CR1 + 2*NI1 - 1.0) + 8.3145*T*(1.0/(-CR1 - NI1 + 1.0) + 1.0/NI1) - 6.54826*T - 2.0*(11082.1315 - 4.45077*T)*(-CR1 - 2*NI1 + 1.0) - 4.0*(11082.1315 - 4.45077*T)*(-CR1 - NI1 + 1.0) - (-2903.220696*CR1 - 2903.220696*NI1 + 2903.220696)*(CR1 + 2*NI1 - 1.0) - 2903.220696*(-CR1 - NI1 + 1.0)*(CR1 + 2*NI1 - 1.0) + 5806.441392*(-CR1/2 - NI1 + 0.5)**2 + 24108.71
    return fyy_NI1

@ti.func
def cal_f(i:int,T:float,y:float) -> float:
    f = 0.0
    if   i == 0: f = f_LIQUID(T,y[0,0],y[0,1])
    elif i == 1: f = f_FCC_A1(T,y[1,0],y[1,1])
    return f

@ti.func
def cal_fyy(a:int,j:int,T:float,y:float) -> float:
    fyy = 0.0
    if   a == 0: # if sublattice 1 in LIQUID
        if   j == 0: fyy = fyy_LIQUID_CR1(T,y[0,0],y[0,1])
        elif j == 1: fyy = fyy_LIQUID_NI1(T,y[0,0],y[0,1])
    elif a == 1: # if sublattice 1 in FCC_A1
        if   j == 0: fyy = fyy_FCC_A1_CR1(T,y[1,0],y[1,1])
        elif j == 1: fyy = fyy_FCC_A1_NI1(T,y[1,0],y[1,1])
    return fyy

