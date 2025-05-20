import numpy as np
import taichi as ti


solutes     = ['CU', 'MG']
uini_ave    = np.array([0.005, 0.063])
yini_LIQUID = np.array([[0.0056843892633574675, 0.06879229961121734]])
yini_FCC_A1 = np.array([[0.00046749287015978154, 0.024639317671506695]])



import taichi as ti
@ti.func
def GHSERAL(T):
    x = 0.0
    if T >= 273.00 and T < 700.00:
        x =  -7976.15+137.093038*T-24.3671976*T*ti.log(T)-1.884662E-3*T**2-0.877664E-6*T**3+74092*(1/T)**(1)
    elif T >= 700.00 and T < 933.47:
        x =  -11276.24+223.048446*T-38.5844296*T*ti.log(T)+18.531982E-3*T**2-5.764227E-6*T**3+74092*(1/T)**(1)
    elif T >= 933.47 and T < 6000.00:
        x =  -11278.378+188.684153*T-31.748192*T*ti.log(T)-1.231E+28*(1/T)**(9)
    return x

@ti.func
def GHSERCU(T):
    x = 0.0
    if T >= 273.00 and T < 1357.77:
        x =  -7770.458+130.485235*T-24.112392*T*ti.log(T)-2.65684E-03*T**2+0.129223E-06*T**3+52478*(1/T)**(1)
    elif T >= 1357.77 and T < 6000.00:
        x =  -13542.026+183.803828*T-31.38*T*ti.log(T)+3.642E+29*(1/T)**(9)
    return x

@ti.func
def GHSERMG(T):
    x = 0.0
    if T >= 273.00 and T < 923.00:
        x =  -8367.34+143.675547*T-26.1849782*T*ti.log(T)+0.4858E-3*T**2-1.393669E-6*T**3+78950*(1/T)**(1)
    elif T >= 923.00 and T < 6000.00:
        x =  -14130.185+204.716215*T-34.3088*T*ti.log(T)+1038.192E25*(1/T)**(9)
    return x

@ti.func
def UFALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  -500+0.45185*T
    return x

@ti.func
def L0FALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +3305.25-2.4*T
    return x

@ti.func
def L1FALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +56.25+0.025*T
    return x

@ti.func
def L2FALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  -325+0.16945*T
    return x

@ti.func
def GFAL3MG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +3*UFALMG(T)
    return x

@ti.func
def GFAL2MG2(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +4*UFALMG(T)
    return x

@ti.func
def GFALMG3(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +3*UFALMG(T)
    return x

@ti.func
def SFALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  -1000+1*UFALMG(T)
    return x

@ti.func
def LDF0ALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +1*GFAL3MG(T)+1.5*GFAL2MG2(T)+1*GFALMG3(T)+1.5*SFALMG(T)+4*L0FALMG(T)
    return x

@ti.func
def LDF1ALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +2*GFAL3MG(T)-2*GFALMG3(T)+4*L1FALMG(T)
    return x

@ti.func
def LDF2ALMG(T):
    x = 0.0
    if T >= 273.00 and T < 6000.00:
        x =  +1*GFAL3MG(T)-1.5*GFAL2MG2(T)+1*GFALMG3(T)-1.5*SFALMG(T)+4*L2FALMG(T)
    return x

@ti.func
def GLIQUIDAL0(T):
    x = 0.0
    if T >= 273.00 and T < 700.00:
        x =  +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL(T)
    elif T >= 700.00 and T < 933.47:
        x =  +11005.03-11.841867*T+7.9337E-20*T**7+GHSERAL(T)
    elif T >= 933.47 and T < 6000.00:
        x =  +10482.382-11.253974*T+1.231E+28*(1/T)**(9)+GHSERAL(T)
    return x

@ti.func
def GLIQUIDCU0(T):
    x = 0.0
    if T >= 273.00 and T < 1357.77:
        x =  +12964.736-9.511904*T-5.849E-21*T**7+GHSERCU(T)
    elif T >= 1357.77 and T < 6000.00:
        x =  +13495.481-9.922344*T-3.642E+29*(1/T)**(9)+GHSERCU(T)
    return x

@ti.func
def GLIQUIDMG0(T):
    x = 0.0
    if T >= 273.00 and T < 923.00:
        x =  +8202.243-8.83693*T+GHSERMG(T)-8.0176E-20*T**7
    elif T >= 923.00 and T < 6000.00:
        x =  +8690.316-9.392158*T+GHSERMG(T)-1.038192E+28*(1/T)**(9)
    return x

@ti.func
def f_LIQUID(T:float,CU1:float,MG1:float) -> float:
    AL1 = (1.0-CU1-MG1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CU1)*ti.log(CU1) +(MG1)*ti.log(MG1) ) )+ \
           + AL1                                       *GLIQUIDAL0(T)+ \
           + CU1                                       *GLIQUIDCU0(T)+ \
           + MG1                                       *GLIQUIDMG0(T)+ \
           + AL1*CU1                                   *(-66054+8.363*T)+ \
           + AL1*CU1*(AL1-CU1)                         *(+32489-8.524*T)+ \
           + AL1*CU1*(AL1-CU1)**2                      *(+7420-10.702*T)+ \
           + AL1*MG1                                   *(-12000.0+8.566*T)+ \
           + AL1*MG1*(AL1-MG1)                         *(+1894.0-3.000*T)+ \
           + AL1*MG1*(AL1-MG1)**2                      *(+2000.0)+ \
           + CU1*MG1                                   *(-36984+4.7561*T)+ \
           + CU1*MG1*(CU1-MG1)                         *(-8191.29)+ \
           + AL1*CU1*MG1*(AL1+(1/3)*(1-(AL1+CU1+MG1))) *(-20000+20*T)+ \
           + AL1*CU1*MG1*(CU1+(1/3)*(1-(AL1+CU1+MG1))) *(+75000)+ \
           + AL1*CU1*MG1*(MG1+(1/3)*(1-(AL1+CU1+MG1))) *(+75000) )

@ti.func
def fyy_LIQUID_CU1(T:float,CU1:float,MG1:float) -> float:
    fyy_CU1 = 2.0*CU1*MG1*(20*T - 20000) - 300000.0*CU1*MG1 + 8.0*CU1*(7420 - 10.702*T)*(-CU1 - MG1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(2*CU1 + MG1 - 1.0) + 4.0*CU1*(32489 - 8.524*T) - 150000.0*MG1**2 + 2.0*MG1*(1894.0 - 3.0*T) + 2.0*MG1*(20*T - 20000)*(2*CU1 + 2*MG1 - 2.0) + 154000.0*MG1*(-CU1 - MG1 + 1.0) - 16000.0*MG1*(CU1/2 + MG1 - 0.5) - 16382.58*MG1 + 8.3145*T*(1.0/(-CU1 - MG1 + 1.0) + 1.0/CU1) - 16.726*T + 4.0*(7420 - 10.702*T)*(-CU1 - MG1 + 1.0)*(2*CU1 + MG1 - 1.0) - 4.0*(7420 - 10.702*T)*(-CU1 - MG1/2 + 0.5)**2 + (29680.0 - 42.808*T)*(-CU1 - MG1 + 1.0)*(2*CU1 + MG1 - 1.0) - (29680.0 - 42.808*T)*(-CU1 - MG1/2 + 0.5)**2 - 1.0*(32489 - 8.524*T)*(-2*CU1 - MG1 + 1.0) - 2.0*(32489 - 8.524*T)*(-CU1 - MG1 + 1.0) - (32489.0 - 8.524*T)*(-2*CU1 - MG1 + 1.0) - 2*(32489.0 - 8.524*T)*(-CU1 - MG1 + 1.0) + 132108.0
    return fyy_CU1
@ti.func
def fyy_LIQUID_MG1(T:float,CU1:float,MG1:float) -> float:
    fyy_MG1 = -150000.0*CU1**2 + 2.0*CU1*MG1*(20*T - 20000) - 300000.0*CU1*MG1 + 2.0*CU1*(7420 - 10.702*T)*(-CU1 - MG1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CU1 + MG1/2 - 0.5) + 2.0*CU1*(32489 - 8.524*T) + 2.0*CU1*(20*T - 20000)*(2*CU1 + 2*MG1 - 2.0) + 150000.0*CU1*(-CU1 - MG1 + 1.0) + 16382.58*CU1 + 4.0*MG1*(1894.0 - 3.0*T) + 16000.0*MG1*(-CU1 - MG1 + 1.0) - 16000.0*MG1*(CU1 + 2*MG1 - 1.0) + 8.3145*T*(1.0/(-CU1 - MG1 + 1.0) + 1.0/MG1) - 17.132*T - 2.0*(1894.0 - 3.0*T)*(-CU1 - 2*MG1 + 1.0) - 4.0*(1894.0 - 3.0*T)*(-CU1 - MG1 + 1.0) + (-8000.0*CU1 - 8000.0*MG1 + 8000.0)*(CU1 + 2*MG1 - 1.0) + 8000.0*(-CU1 - MG1 + 1.0)*(CU1 + 2*MG1 - 1.0) - 16000.0*(-CU1/2 - MG1 + 0.5)**2 + 24000.0
    return fyy_MG1

@ti.func
def f_FCC_A1(T:float,CU1:float,MG1:float) -> float:
    AL1 = (1.0-CU1-MG1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CU1)*ti.log(CU1) +(MG1)*ti.log(MG1) ) )+ \
           + AL1                                       *(+GHSERAL(T))+ \
           + CU1                                       *(+GHSERCU(T))+ \
           + MG1                                       *(+2600-0.90*T+GHSERMG(T))+ \
           + AL1*CU1                                   *(-53520+2*T)+ \
           + AL1*CU1*(AL1-CU1)                         *(+38590-2*T)+ \
           + AL1*CU1*(AL1-CU1)**2                      *(+1170)+ \
           + AL1*CU1*(AL1-CU1)**3                      *(+1E-8)+ \
           + AL1*MG1                                   *(+1*LDF0ALMG(T))+ \
           + AL1*MG1*(AL1-MG1)                         *(+1*LDF1ALMG(T))+ \
           + AL1*MG1*(AL1-MG1)**2                      *(+1*LDF2ALMG(T))+ \
           + CU1*MG1                                   *(-22279.28+5.868*T)+ \
           + AL1*CU1*MG1*(AL1+(1/3)*(1-(AL1+CU1+MG1))) *(+60000)+ \
           + AL1*CU1*MG1*(CU1+(1/3)*(1-(AL1+CU1+MG1))) *(+1E-8)+ \
           + AL1*CU1*MG1*(MG1+(1/3)*(1-(AL1+CU1+MG1))) *(+1E-8) )

@ti.func
def fyy_FCC_A1_CU1(T:float,CU1:float,MG1:float) -> float:
    fyy_CU1 = 119999.99999996*CU1*MG1 + 4.0*CU1*(38590 - 2*T) - 2.4e-7*CU1*(-CU1 - MG1 + 1.0)*(2*CU1 + MG1 - 1.0) + 9360.0*CU1*(-CU1 - MG1 + 1.0) + 4.8e-7*CU1*(-CU1 - MG1/2 + 0.5)**2 - 9360.0*CU1*(2*CU1 + MG1 - 1.0) - 2.0e-8*MG1**2 + 2.0*MG1*(-CU1 - MG1 + 1.0)*LDF2ALMG(T) + 2.0e-8*MG1*(-CU1 - MG1 + 1.0) - 8.0*MG1*(CU1/2 + MG1 - 0.5)*LDF2ALMG(T) + 120000.0*MG1*(2*CU1 + 2*MG1 - 2.0) + 2.0*MG1*LDF1ALMG(T) + 8.3145*T*(1.0/(-CU1 - MG1 + 1.0) + 1.0/CU1) - 4.0*T - (38590.0 - 2.0*T)*(-2*CU1 - MG1 + 1.0) - 2*(38590.0 - 2.0*T)*(-CU1 - MG1 + 1.0) - 1.0*(38590 - 2*T)*(-2*CU1 - MG1 + 1.0) - 2.0*(38590 - 2*T)*(-CU1 - MG1 + 1.0) + (-4680.0*CU1 - 4680.0*MG1 + 4680.0)*(2*CU1 + MG1 - 1.0) - 2.4e-7*(-CU1 - MG1 + 1.0)*(-CU1 - MG1/2 + 0.5)**2 + 4680.0*(-CU1 - MG1 + 1.0)*(2*CU1 + MG1 - 1.0) - 1.6e-7*(-CU1 - MG1/2 + 0.5)**3 - 3*(-CU1 - MG1/2 + 0.5)**2*(-8.0e-8*CU1 - 8.0e-8*MG1 + 8.0e-8) - 9360.0*(-CU1 - MG1/2 + 0.5)**2 + 107040.0
    return fyy_CU1
@ti.func
def fyy_FCC_A1_MG1(T:float,CU1:float,MG1:float) -> float:
    fyy_MG1 = -2.0e-8*CU1**2 + 119999.99999996*CU1*MG1 + 2.0*CU1*(38590 - 2*T) - 1.2e-7*CU1*(-CU1 - MG1 + 1.0)*(CU1 + MG1/2 - 0.5) + 2340.00000002*CU1*(-CU1 - MG1 + 1.0) + 2.4e-7*CU1*(-CU1 - MG1/2 + 0.5)**2 - 9360.0*CU1*(CU1 + MG1/2 - 0.5) + 120000.0*CU1*(2*CU1 + 2*MG1 - 2.0) + 8.0*MG1*(-CU1 - MG1 + 1.0)*LDF2ALMG(T) - 8.0*MG1*(CU1 + 2*MG1 - 1.0)*LDF2ALMG(T) + 4.0*MG1*LDF1ALMG(T) + 8.3145*T*(1.0/(-CU1 - MG1 + 1.0) + 1.0/MG1) + (-4.0*CU1 - 4.0*MG1 + 4.0)*(CU1 + 2*MG1 - 1.0)*LDF2ALMG(T) - 1.0*(-CU1 - 2*MG1 + 1.0)*LDF1ALMG(T) - (-1.0*CU1 - 2.0*MG1 + 1.0)*LDF1ALMG(T) + 4.0*(-CU1 - MG1 + 1.0)*(CU1 + 2*MG1 - 1.0)*LDF2ALMG(T) - 4.0*(-CU1 - MG1 + 1.0)*LDF1ALMG(T) - 8.0*(-CU1/2 - MG1 + 0.5)**2*LDF2ALMG(T) - 2.0*LDF0ALMG(T)
    return fyy_MG1

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
        if   j == 0: fyy = fyy_LIQUID_CU1(T,y[0,0],y[0,1])
        elif j == 1: fyy = fyy_LIQUID_MG1(T,y[0,0],y[0,1])
    elif a == 1: # if sublattice 1 in FCC_A1
        if   j == 0: fyy = fyy_FCC_A1_CU1(T,y[1,0],y[1,1])
        elif j == 1: fyy = fyy_FCC_A1_MG1(T,y[1,0],y[1,1])
    return fyy

