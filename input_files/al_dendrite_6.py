import numpy as np
import taichi as ti


solutes     = ['CR', 'CU', 'FE', 'MG', 'MN']
uini_ave    = np.array([0.005, 0.038, 0.005, 0.005, 0.005])
yini_LIQUID = np.array([[0.003843902114302722, 0.04336020966304083, 0.0057588986088385336, 0.005525448099756957, 0.005200237733369166]])
yini_FCC_A1 = np.array([[0.012252651821312844, 0.004373319368433703, 0.00023913377361450459, 0.0017036596422203162, 0.003743830795227265]])

@ti.func
def GHSERAL(T): return -11276.24+223.048446*T-38.5844296*T*ti.log(T)+18.531982E-3*T**2-5.764227E-6*T**3+74092*(1/T)**(1)
@ti.func
def GHSERCR(T): return -8856.94+157.48*T-26.908*T*ti.log(T)+0.00189435*T**2-1.47721E-6*T**3+139250*(1/T)**(1)
@ti.func
def GHSERCU(T): return -7770.458+130.485235*T-24.112392*T*ti.log(T)-2.65684E-03*T**2+0.129223E-06*T**3+52478*(1/T)**(1)
@ti.func
def GHSERFE(T): return +1225.7+124.134*T-23.5143*T*ti.log(T)-0.00439752*T**2-5.89269E-8*T**3+77358.5*(1/T)**(1)
@ti.func
def GHSERMG(T): return -8367.34+143.675547*T-26.1849782*T*ti.log(T)+0.4858E-3*T**2-1.393669E-6*T**3+78950*(1/T)**(1)
@ti.func
def GHSERMN(T): return -8115.28+130.059*T-23.4582*T*ti.log(T)-7.34768E-3*T**2+69827.1*(1/T)**(1)
@ti.func
def GHSERNI(T): return -5179.159+117.854*T-22.096*T*ti.log(T)-4.8407E-3*T**2
@ti.func
def GHSERSI(T): return -8162.609+137.236859*T-22.8317533*T*ti.log(T)-1.912904E-3*T**2-0.003552E-6*T**3+176667*(1/T)**(1)
@ti.func
def GHSERTI(T): return -8059.921+133.615208*T-23.9933*T*ti.log(T)-4.777975E-3*T**2+1.06716E-07*T**3+72636*(1/T)**(1)
@ti.func
def GHSERZN(T): return -11070.559+172.34566*T-31.38*T*ti.log(T)+470.514E24*(1/T)**(9)
@ti.func
def GHSERZR(T): return -7827.595+125.64905*T-24.1618*T*ti.log(T)-4.37791E-3*T**2+34971*(1/T)**(1)
@ti.func
def GSCFCC(T): return -2511.295+132.759582*T-24.9132*T*ti.log(T)-5.73295E-04*T**2-8.59345E-07*T**3
@ti.func
def UFALMG(T): return -500+0.45185*T
@ti.func
def L0FALMG(T): return +3305.25-2.4*T
@ti.func
def L1FALMG(T): return +56.25+0.025*T
@ti.func
def L2FALMG(T): return -325+0.16945*T
@ti.func
def GFAL3MG(T): return +3*UFALMG(T)
@ti.func
def GFAL2MG2(T): return +4*UFALMG(T)
@ti.func
def GFALMG3(T): return +3*UFALMG(T)
@ti.func
def SFALMG(T): return -1000+1*UFALMG(T)
@ti.func
def LDF0ALMG(T): return +1*GFAL3MG(T)+1.5*GFAL2MG2(T)+1*GFALMG3(T)+1.5*SFALMG(T)+4*L0FALMG(T)
@ti.func
def LDF1ALMG(T): return +2*GFAL3MG(T)-2*GFALMG3(T)+4*L1FALMG(T)
@ti.func
def LDF2ALMG(T): return +1*GFAL3MG(T)-1.5*GFAL2MG2(T)+1*GFALMG3(T)-1.5*SFALMG(T)+4*L2FALMG(T)
@ti.func
def UFALSI(T): return -8000
@ti.func
def GFAL3SI(T): return +3750+3*UFALSI(T)
@ti.func
def GFAL2SI2(T): return +4*UFALSI(T)
@ti.func
def GFALSI3(T): return +5000+3*UFALSI(T)
@ti.func
def SFALSI(T): return +UFALSI(T)
@ti.func
def LDF0ALSI(T): return +95691.09-0.09584*T+1*GFAL3SI(T)+1.5*GFAL2SI2(T)+1*GFALSI3(T)+1.5*SFALSI(T)
@ti.func
def LDF1ALSI(T): return +2500+2*GFAL3SI(T)-2*GFALSI3(T)
@ti.func
def LDF2ALSI(T): return -20735+1*GFAL3SI(T)-1.5*GFAL2SI2(T)+1*GFALSI3(T)-1.5*SFALSI(T)
@ti.func
def UFSIMG(T): return -16000
@ti.func
def GFSI3MG(T): return +2000+3*UFSIMG(T)
@ti.func
def GFSI2MG2(T): return +4*UFSIMG(T)
@ti.func
def GFSIMG3(T): return +3*UFSIMG(T)
@ti.func
def SFSIMG(T): return +UFSIMG(T)
@ti.func
def LDF0SIMG(T): return +200000+0.89361*T+1*GFSI3MG(T)+1.5*GFSI2MG2(T)+1*GFSIMG3(T)+1.5*SFSIMG(T)
@ti.func
def LDF1SIMG(T): return +2*GFSI3MG(T)-2*GFSIMG3(T)
@ti.func
def LDF2SIMG(T): return -24000+1*GFSI3MG(T)-1.5*GFSI2MG2(T)+1*GFSIMG3(T)-1.5*SFSIMG(T)
@ti.func
def GSCLIQ(T): return +6478.66+45.427539*T-10.7967803*T*ti.log(T)-20.636524E-3*T**2+2.13106E-6*T**3-158106*(1/T)**(1)
@ti.func
def GZRLIQ(T): return +18147.703-9.080762*T+1.6275E-22*T**7+GHSERZR(T)
@ti.func
def GLIQUIDAL0(T): return +11005.03-11.841867*T+7.9337E-20*T**7+GHSERAL(T)
@ti.func
def GLIQUIDCR0(T): return +24339.955-11.420225*T+2.37615E-21*T**7+GHSERCR(T)
@ti.func
def GLIQUIDCU0(T): return +12964.736-9.511904*T-5.849E-21*T**7+GHSERCU(T)
@ti.func
def GLIQUIDFE0(T): return +12040.17-6.55843*T-3.6751551E-21*T**7+GHSERFE(T)
@ti.func
def GLIQUIDMG0(T): return +8202.243-8.83693*T+GHSERMG(T)-8.0176E-20*T**7
@ti.func
def GLIQUIDMN0(T): return +17859.91-12.6208*T-4.419297E-21*T**7+GHSERMN(T)
@ti.func
def GLIQUIDNI0(T): return +16414.686-9.397*T-3.82318E-21*T**7+GHSERNI(T)
@ti.func
def GLIQUIDSI0(T): return +50696.36-30.099439*T+2.0931E-21*T**7+GHSERSI(T)
@ti.func
def GLIQUIDTI0(T): return +12194.415-6.980938*T+GHSERTI(T)
@ti.func
def GLIQUIDZN0(T): return -3620.391+161.608594*T-31.38*T*ti.log(T)
@ti.func
def GFCC_A1FEVA0(T): return -1462.4+8.282*T-1.15*T*ti.log(T)+0.00064*T**2+GHSERFE(T)
@ti.func
def GFCC_A1MNVA0(T): return -3439.3+131.884*T-24.5177*T*ti.log(T)-0.006*T**2+69600*(1/T)**(1)



@ti.func
def f_LIQUID(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    AL1 = (1.0-CR1-CU1-FE1-MG1-MN1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CR1)*ti.log(CR1) +(CU1)*ti.log(CU1) +(FE1)*ti.log(FE1) +(MG1)*ti.log(MG1) +(MN1)*ti.log(MN1) ) )+ \
           + AL1                                       *GLIQUIDAL0(T)+ \
           + CR1                                       *GLIQUIDCR0(T)+ \
           + CU1                                       *GLIQUIDCU0(T)+ \
           + FE1                                       *GLIQUIDFE0(T)+ \
           + MG1                                       *GLIQUIDMG0(T)+ \
           + MN1                                       *GLIQUIDMN0(T)+ \
           + AL1*CR1                                   *(-37139.1+2.0110*T)+ \
           + AL1*CR1*(AL1-CR1)                         *(-15698.7+7.4555*T)+ \
           + AL1*CU1                                   *(-66054+8.363*T)+ \
           + AL1*CU1*(AL1-CU1)                         *(+32489-8.524*T)+ \
           + AL1*CU1*(AL1-CU1)**2                      *(+7420-10.702*T)+ \
           + AL1*FE1                                   *(-91976.5+22.1314*T)+ \
           + AL1*FE1*(AL1-FE1)                         *(-5672.58+4.8728*T)+ \
           + AL1*FE1*(AL1-FE1)**2                      *(+121.9)+ \
           + AL1*MG1                                   *(-12000.0+8.566*T)+ \
           + AL1*MG1*(AL1-MG1)                         *(+1894.0-3.000*T)+ \
           + AL1*MG1*(AL1-MG1)**2                      *(+2000.0)+ \
           + AL1*MN1                                   *(-70584.9+28.22693*T)+ \
           + AL1*MN1*(AL1-MN1)                         *(-13293.5+9.82551*T)+ \
           + CR1*CU1                                   *(+83730-105.12*T+10*T*ti.log(T))+ \
           + CR1*CU1*(CR1-CU1)                         *(-1371.45)+ \
           + CR1*CU1*(CR1-CU1)**2                      *(-1271.47)+ \
           + CR1*FE1                                   *(-17737+7.996546*T)+ \
           + CR1*FE1*(CR1-FE1)                         *(-1331)+ \
           + CR1*MG1                                   *(+94500.0)+ \
           + CR1*MG1*(CR1-MG1)                         *(+12500.0)+ \
           + CR1*MN1                                   *(-15009+13.6587*T)+ \
           + CR1*MN1*(CR1-MN1)                         *(+504+0.9479*T)+ \
           + CU1*FE1                                   *(+73316.72-142.79*T+15.82*T*ti.log(T))+ \
           + CU1*FE1*(CU1-FE1)                         *(+9100.15-5.94*T)+ \
           + CU1*FE1*(CU1-FE1)**2                      *(+2428.96)+ \
           + CU1*FE1*(CU1-FE1)**3                      *(-233.62)+ \
           + CU1*MG1                                   *(-36984+4.7561*T)+ \
           + CU1*MG1*(CU1-MG1)                         *(-8191.29)+ \
           + CU1*MN1                                   *(+1118.55-5.6225*T)+ \
           + CU1*MN1*(CU1-MN1)                         *(-10915.375)+ \
           + FE1*MG1                                   *(+61343.0+1.5*T)+ \
           + FE1*MG1*(FE1-MG1)                         *(-2700.0)+ \
           + FE1*MN1                                   *(-3950+0.489*T)+ \
           + FE1*MN1*(FE1-MN1)                         *(+1145)+ \
           + MG1*MN1                                   *(+19125.0+12.5*T)+ \
           + AL1*CR1*MG1                               *(-30000)+ \
           + AL1*CR1*CU1                               *(-30000)+ \
           + AL1*CR1*MN1                               *(-240000)+ \
           + AL1*CU1*MG1*(AL1+(1/3)*(1-(AL1+CU1+MG1))) *(-20000+20*T)+ \
           + AL1*CU1*MG1*(CU1+(1/3)*(1-(AL1+CU1+MG1))) *(+75000)+ \
           + AL1*CU1*MG1*(MG1+(1/3)*(1-(AL1+CU1+MG1))) *(+75000)+ \
           + AL1*CU1*MN1                               *(+25000)+ \
           + AL1*FE1*MG1                               *(-100000)+ \
           + AL1*FE1*MN1                               *(-200000)+ \
           + AL1*MG1*MN1*(AL1+(1/3)*(1-(AL1+MG1+MN1))) *(+28000)+ \
           + AL1*MG1*MN1*(MG1+(1/3)*(1-(AL1+MG1+MN1))) *(+1E-8)+ \
           + AL1*MG1*MN1*(MN1+(1/3)*(1-(AL1+MG1+MN1))) *(+1E-8)+ \
           + CR1*CU1*FE1*(CR1+(1/3)*(1-(CR1+CU1+FE1))) *(-115799+61.673*T)+ \
           + CR1*CU1*FE1*(CU1+(1/3)*(1-(CR1+CU1+FE1))) *(-89317+55.011*T)+ \
           + CR1*CU1*FE1*(FE1+(1/3)*(1-(CR1+CU1+FE1))) *(+116631-57.980*T)+ \
           + CR1*FE1*MN1                               *(+2378)+ \
           + CU1*FE1*MN1*(CU1+(1/3)*(1-(CU1+FE1+MN1))) *(+241298-140*T)+ \
           + CU1*FE1*MN1*(FE1+(1/3)*(1-(CU1+FE1+MN1))) *(+82096-40*T)+ \
           + CU1*FE1*MN1*(MN1+(1/3)*(1-(CU1+FE1+MN1))) *(+25901.5-20*T) )

@ti.func
def fyy_LIQUID_CR1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_CR1 = -2542.94*CR1*CU1 + 4.0*CR1*(7.4555*T - 15698.7) - 0.666666666666666*CU1*FE1*(116631 - 57.98*T) - 0.666666666666666*CU1*FE1*(55.011*T - 89317) + 1.33333333333333*CU1*FE1*(61.673*T - 115799) + 1.33333333333333*CU1*MG1*(20*T - 20000) - 100000.0*CU1*MG1 + 2.0*CU1*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2.0*CU1*(32489 - 8.524*T) - 2542.94*CU1*(2*CR1 - 2*CU1) + 57257.1*CU1 + 2.0*FE1*(4.8728*T - 5672.58) + 243.8*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 975.2*FE1*(CR1/2 + CU1/2 + FE1 + MG1/2 + MN1/2 - 0.5) - 2662.0*FE1 + 37333.33333332*MG1*MN1 + 2.0*MG1*(1894.0 - 3.0*T) + 4000.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 16000.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5) + 85000.0*MG1 + 2.0*MN1*(0.9479*T + 504) + 2.0*MN1*(9.82551*T - 13293.5) + 480000.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/CR1) - 4.022*T - 2.0*(7.4555*T - 15698.7)*(-2*CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 4.0*(7.4555*T - 15698.7)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 74278.2
    return fyy_CR1
@ti.func
def fyy_LIQUID_CU1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_CU1 = -2542.94*CR1*CU1 - 0.666666666666666*CR1*FE1*(116631 - 57.98*T) + 1.33333333333333*CR1*FE1*(55.011*T - 89317) - 0.666666666666666*CR1*FE1*(61.673*T - 115799) - 2542.94*CR1*(-2*CR1 + 2*CU1) + 2.0*CR1*(7.4555*T - 15698.7) + 62742.9*CR1 - 700.86*CU1*FE1*(2*CU1 - 2*FE1) + 4857.92*CU1*FE1 + 2.0*CU1*MG1*(20*T - 20000) - 150000.0*CU1*MG1 + 8.0*CU1*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) + 4.0*CU1*(32489 - 8.524*T) - 0.666666666666666*FE1*MN1*(25901.5 - 20*T) - 0.666666666666666*FE1*MN1*(82096 - 40*T) + 1.33333333333333*FE1*MN1*(241298 - 140*T) + 2.0*FE1*(9100.15 - 5.94*T) - 1401.72*FE1*(CU1 - FE1)**2 + 4857.92*FE1*(2*CU1 - 2*FE1) + 2.0*FE1*(4.8728*T - 5672.58) + 243.8*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 975.2*FE1*(CR1/2 + CU1/2 + FE1 + MG1/2 + MN1/2 - 0.5) + 37333.33333332*MG1*MN1 + 2.0*MG1*(1894.0 - 3.0*T) - 2.0*MG1*(20*T - 20000)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 2.0*MG1*(20*T - 20000)*(-2*CR1/3 - CU1 - 2*FE1/3 - MG1 - 2*MN1/3 + 1.0) - 150000.0*MG1*(CR1/3 + CU1 + FE1/3 + MN1/3) - 150000.0*MG1*(CR1/3 + FE1/3 + MG1 + MN1/3) + 154000.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 16000.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5) - 16382.58*MG1 + 2.0*MN1*(9.82551*T - 13293.5) - 71830.75*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/CU1) - 16.726*T + 4.0*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) - 4.0*(7420 - 10.702*T)*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 + (29680.0 - 42.808*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) - (29680.0 - 42.808*T)*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 1.0*(32489 - 8.524*T)*(-CR1 - 2*CU1 - FE1 - MG1 - MN1 + 1.0) - 2.0*(32489 - 8.524*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - (32489.0 - 8.524*T)*(-CR1 - 2*CU1 - FE1 - MG1 - MN1 + 1.0) - 2*(32489.0 - 8.524*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 132108.0
    return fyy_CU1
@ti.func
def fyy_LIQUID_FE1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_FE1 = 1.33333333333333*CR1*CU1*(116631 - 57.98*T) - 0.666666666666666*CR1*CU1*(55.011*T - 89317) - 0.666666666666666*CR1*CU1*(61.673*T - 115799) + 2.0*CR1*(7.4555*T - 15698.7) + 2662.0*CR1 + 700.86*CU1*FE1*(-2*CU1 + 2*FE1) + 4857.92*CU1*FE1 + 1.33333333333333*CU1*MG1*(20*T - 20000) - 100000.0*CU1*MG1 - 0.666666666666666*CU1*MN1*(25901.5 - 20*T) + 1.33333333333333*CU1*MN1*(82096 - 40*T) - 0.666666666666666*CU1*MN1*(241298 - 140*T) + 2.0*CU1*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) - 2.0*CU1*(9100.15 - 5.94*T) + 2.0*CU1*(32489 - 8.524*T) + 4857.92*CU1*(-2*CU1 + 2*FE1) + 1401.72*CU1*(CU1 - FE1)**2 + 4.0*FE1*(4.8728*T - 5672.58) + 975.2*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 975.2*FE1*(CR1 + CU1 + 2*FE1 + MG1 + MN1 - 1.0) + 37333.33333332*MG1*MN1 + 2.0*MG1*(1894.0 - 3.0*T) + 4000.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 16000.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5) + 194600.0*MG1 + 2.0*MN1*(9.82551*T - 13293.5) + 402290.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/FE1) - 44.2628*T - 2.0*(4.8728*T - 5672.58)*(-CR1 - CU1 - 2*FE1 - MG1 - MN1 + 1.0) - 4.0*(4.8728*T - 5672.58)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + (-487.6*CR1 - 487.6*CU1 - 487.6*FE1 - 487.6*MG1 - 487.6*MN1 + 487.6)*(CR1 + CU1 + 2*FE1 + MG1 + MN1 - 1.0) + 487.6*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + CU1 + 2*FE1 + MG1 + MN1 - 1.0) - 975.2*(-CR1/2 - CU1/2 - FE1 - MG1/2 - MN1/2 + 0.5)**2 + 183953.0
    return fyy_FE1
@ti.func
def fyy_LIQUID_MG1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_MG1 = 2.0*CR1*(7.4555*T - 15698.7) + 35000.0*CR1 + 2.0*CU1*MG1*(20*T - 20000) - 150000.0*CU1*MG1 + 2.0*CU1*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2.0*CU1*(32489 - 8.524*T) - 2.0*CU1*(20*T - 20000)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 2.0*CU1*(20*T - 20000)*(-2*CR1/3 - CU1 - 2*FE1/3 - MG1 - 2*MN1/3 + 1.0) - 150000.0*CU1*(CR1/3 + CU1 + FE1/3 + MN1/3) - 150000.0*CU1*(CR1/3 + FE1/3 + MG1 + MN1/3) + 150000.0*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 16382.58*CU1 + 2.0*FE1*(4.8728*T - 5672.58) + 243.8*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 975.2*FE1*(CR1/2 + CU1/2 + FE1 + MG1/2 + MN1/2 - 0.5) + 205400.0*FE1 + 55999.99999998*MG1*MN1 + 4.0*MG1*(1894.0 - 3.0*T) + 16000.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 16000.0*MG1*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0) + 2.0*MN1*(9.82551*T - 13293.5) - 2.0e-8*MN1*(CR1/3 + CU1/3 + FE1/3 + MG1) - 2.0e-8*MN1*(CR1/3 + CU1/3 + FE1/3 + MN1) - 55999.99999998*MN1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 56000.0*MN1*(-2*CR1/3 - 2*CU1/3 - 2*FE1/3 - MG1 - MN1 + 1.0) + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/MG1) - 17.132*T - 2.0*(1894.0 - 3.0*T)*(-CR1 - CU1 - FE1 - 2*MG1 - MN1 + 1.0) - 4.0*(1894.0 - 3.0*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + (-8000.0*CR1 - 8000.0*CU1 - 8000.0*FE1 - 8000.0*MG1 - 8000.0*MN1 + 8000.0)*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0) + 8000.0*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0) - 16000.0*(-CR1/2 - CU1/2 - FE1/2 - MG1 - MN1/2 + 0.5)**2 + 24000.0
    return fyy_MG1
@ti.func
def fyy_LIQUID_MN1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_MN1 = -2.0*CR1*(0.9479*T + 504) + 2.0*CR1*(7.4555*T - 15698.7) + 480000.0*CR1 + 1.33333333333333*CU1*FE1*(25901.5 - 20*T) - 0.666666666666666*CU1*FE1*(82096 - 40*T) - 0.666666666666666*CU1*FE1*(241298 - 140*T) + 1.33333333333333*CU1*MG1*(20*T - 20000) - 100000.0*CU1*MG1 + 2.0*CU1*(7420 - 10.702*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 8.0*CU1*(7420 - 10.702*T)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2.0*CU1*(32489 - 8.524*T) - 28169.25*CU1 + 2.0*FE1*(4.8728*T - 5672.58) + 243.8*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 975.2*FE1*(CR1/2 + CU1/2 + FE1 + MG1/2 + MN1/2 - 0.5) + 397710.0*FE1 + 55999.99999998*MG1*MN1 + 2.0*MG1*(1894.0 - 3.0*T) - 2.0e-8*MG1*(CR1/3 + CU1/3 + FE1/3 + MG1) - 2.0e-8*MG1*(CR1/3 + CU1/3 + FE1/3 + MN1) - 51999.99999998*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 56000.0*MG1*(-2*CR1/3 - 2*CU1/3 - 2*FE1/3 - MG1 - MN1 + 1.0) - 16000.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5) + 4.0*MN1*(9.82551*T - 13293.5) + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/MN1) - 56.45386*T - 2.0*(9.82551*T - 13293.5)*(-CR1 - CU1 - FE1 - MG1 - 2*MN1 + 1.0) - 4.0*(9.82551*T - 13293.5)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 141169.8
    return fyy_MN1

@ti.func
def f_FCC_A1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    AL1 = (1.0-CR1-CU1-FE1-MG1-MN1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CR1)*ti.log(CR1) +(CU1)*ti.log(CU1) +(FE1)*ti.log(FE1) +(MG1)*ti.log(MG1) +(MN1)*ti.log(MN1) ) )+ \
           + AL1                                       *(+GHSERAL(T))+ \
           + CR1                                       *(+7284+0.163*T+GHSERCR(T))+ \
           + CU1                                       *(+GHSERCU(T))+ \
           + FE1                                       *GFCC_A1FEVA0(T)+ \
           + MG1                                       *(+2600-0.90*T+GHSERMG(T))+ \
           + MN1                                       *GFCC_A1MNVA0(T)+ \
           + AL1*CR1                                   *(-49516.6+7.3288*T)+ \
           + AL1*CU1                                   *(-53520+2*T)+ \
           + AL1*CU1*(AL1-CU1)                         *(+38590-2*T)+ \
           + AL1*CU1*(AL1-CU1)**2                      *(+1170)+ \
           + AL1*CU1*(AL1-CU1)**3                      *(+1E-8)+ \
           + AL1*FE1                                   *(-76066.1+18.6758*T)+ \
           + AL1*FE1*(AL1-FE1)                         *(+21167.4+1.3398*T)+ \
           + AL1*MG1                                   *(+1*LDF0ALMG(T))+ \
           + AL1*MG1*(AL1-MG1)                         *(+1*LDF1ALMG(T))+ \
           + AL1*MG1*(AL1-MG1)**2                      *(+1*LDF2ALMG(T))+ \
           + AL1*MN1                                   *(-69300+25.0000*T)+ \
           + AL1*MN1*(AL1-MN1)                         *(+8800.0)+ \
           + CR1*CU1                                   *(+67800+6*T)+ \
           + CR1*FE1                                   *(+10833-7.477*T)+ \
           + CR1*FE1*(CR1-FE1)                         *(+1410)+ \
           + CR1*MG1                                   *(+80*T)+ \
           + CR1*MN1                                   *(-19088+17.5423*T)+ \
           + CU1*FE1                                   *(+48885.74-11.51*T)+ \
           + CU1*FE1*(CU1-FE1)                         *(+12687.16-8.01*T)+ \
           + CU1*FE1*(CU1-FE1)**2                      *(+4054.11)+ \
           + CU1*MG1                                   *(-22279.28+5.868*T)+ \
           + CU1*MN1                                   *(+20235.508-13.2437*T)+ \
           + CU1*MN1*(CU1-MN1)                         *(-12154.853+2.9399*T)+ \
           + CU1*MN1*(CU1-MN1)**2                      *(+1037.56)+ \
           + FE1*MG1                                   *(+65200.0)+ \
           + FE1*MN1                                   *(-7762+3.86*T)+ \
           + FE1*MN1*(FE1-MN1)                         *(-259)+ \
           + MG1*MN1                                   *(+70000.0)+ \
           + AL1*CR1*CU1                               *(+1E-6)+ \
           + AL1*CR1*MG1                               *(+1E-6)+ \
           + AL1*CR1*MN1*(AL1+(1/3)*(1-(AL1+CR1+MN1))) *(-50000)+ \
           + AL1*CR1*MN1*(CR1+(1/3)*(1-(AL1+CR1+MN1))) *(+50000)+ \
           + AL1*CR1*MN1*(MN1+(1/3)*(1-(AL1+CR1+MN1))) *(+1E-8)+ \
           + AL1*CU1*MG1*(AL1+(1/3)*(1-(AL1+CU1+MG1))) *(+60000)+ \
           + AL1*CU1*MG1*(CU1+(1/3)*(1-(AL1+CU1+MG1))) *(+1E-8)+ \
           + AL1*CU1*MG1*(MG1+(1/3)*(1-(AL1+CU1+MG1))) *(+1E-8)+ \
           + AL1*CU1*MN1*(AL1+(1/3)*(1-(AL1+CU1+MN1))) *(+15000)+ \
           + AL1*CU1*MN1*(CU1+(1/3)*(1-(AL1+CU1+MN1))) *(+50000)+ \
           + AL1*CU1*MN1*(MN1+(1/3)*(1-(AL1+CU1+MN1))) *(+15000)+ \
           + AL1*FE1*MN1*(AL1+(1/3)*(1-(AL1+FE1+MN1))) *(+0.0001)+ \
           + AL1*FE1*MN1*(FE1+(1/3)*(1-(AL1+FE1+MN1))) *(-63652)+ \
           + AL1*FE1*MN1*(MN1+(1/3)*(1-(AL1+FE1+MN1))) *(-26753)+ \
           + CR1*CU1*FE1*(CR1+(1/3)*(1-(CR1+CU1+FE1))) *(+1E-7)+ \
           + CR1*CU1*FE1*(CU1+(1/3)*(1-(CR1+CU1+FE1))) *(+1E-7)+ \
           + CR1*CU1*FE1*(FE1+(1/3)*(1-(CR1+CU1+FE1))) *(-29976+24.982*T)+ \
           + CR1*FE1*MN1                               *(+6715-10.393*T)+ \
           + CU1*FE1*MN1*(CU1+(1/3)*(1-(CU1+FE1+MN1))) *(-60000+105*T)+ \
           + CU1*FE1*MN1*(FE1+(1/3)*(1-(CU1+FE1+MN1))) *(-121140+89*T)+ \
           + CU1*FE1*MN1*(MN1+(1/3)*(1-(CU1+FE1+MN1))) *(-160467+112*T) )

@ti.func
def fyy_FCC_A1_CR1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_CR1 = -200000.0*CR1*MN1 - 0.666666666666666*CU1*FE1*(24.982*T - 29976) + 6.66666666666666e-8*CU1*FE1 + 79999.9999999866*CU1*MG1 - 23333.3333333334*CU1*MN1 + 2.0*CU1*(38590 - 2*T) - 1.2e-7*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2340.0*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 2.4e-7*CU1*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 9360.0*CU1*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) - 2.0e-6*CU1 + 60270.0001333334*FE1*MN1 + 2.0*FE1*(1.3398*T + 21167.4) + 2820.0*FE1 + 2.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF2ALMG(T) - 8.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5)*LDF2ALMG(T) + 2.0*MG1*LDF1ALMG(T) - 2.0e-6*MG1 - 100000.0*MN1*(CR1 + CU1/3 + FE1/3 + MG1/3) - 2.0e-8*MN1*(CU1/3 + FE1/3 + MG1/3 + MN1) + 200000.0*MN1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 100000.0*MN1*(-CR1 - 2*CU1/3 - 2*FE1/3 - 2*MG1/3 - MN1 + 1.0) + 17600.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/CR1) - 14.6576*T + 99033.2
    return fyy_CR1
@ti.func
def fyy_FCC_A1_CU1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_CU1 = -0.666666666666666*CR1*FE1*(24.982*T - 29976) + 6.66666666666666e-8*CR1*FE1 - 100000.000000007*CR1*MN1 - 2.0e-6*CR1 + 8108.22*CU1*FE1 + 119999.99999998*CU1*MG1 - 67924.88*CU1*MN1 + 4.0*CU1*(38590 - 2*T) - 2.4e-7*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) + 9360.0*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 4.8e-7*CU1*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 9360.0*CU1*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) - 0.666666666666666*FE1*MN1*(89*T - 121140) + 1.33333333333333*FE1*MN1*(105*T - 60000) - 0.666666666666666*FE1*MN1*(112*T - 160467) + 60270.0001333334*FE1*MN1 + 2.0*FE1*(12687.16 - 8.01*T) + 8108.22*FE1*(2*CU1 - 2*FE1) + 2.0*FE1*(1.3398*T + 21167.4) - 2.0e-8*MG1*(CR1/3 + CU1 + FE1/3 + MN1/3) - 2.0e-8*MG1*(CR1/3 + FE1/3 + MG1 + MN1/3) + 2.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF2ALMG(T) - 119999.99999998*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 120000.0*MG1*(-2*CR1/3 - CU1 - 2*FE1/3 - MG1 - 2*MN1/3 + 1.0) - 8.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5)*LDF2ALMG(T) + 2.0*MG1*LDF1ALMG(T) + 2075.12*MN1*(2*CU1 - 2*MN1) + 2.0*MN1*(2.9399*T - 12154.853) - 100000.0*MN1*(CR1/3 + CU1 + FE1/3 + MG1/3) - 30000.0*MN1*(CR1/3 + FE1/3 + MG1/3 + MN1) + 70000.0*MN1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 30000.0*MN1*(-2*CR1/3 - CU1 - 2*FE1/3 - 2*MG1/3 - MN1 + 1.0) + 17600.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/CU1) - 4.0*T - (38590.0 - 2.0*T)*(-CR1 - 2*CU1 - FE1 - MG1 - MN1 + 1.0) - 2*(38590.0 - 2.0*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 1.0*(38590 - 2*T)*(-CR1 - 2*CU1 - FE1 - MG1 - MN1 + 1.0) - 2.0*(38590 - 2*T)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + (-4680.0*CR1 - 4680.0*CU1 - 4680.0*FE1 - 4680.0*MG1 - 4680.0*MN1 + 4680.0)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) - 2.4e-7*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 + 4680.0*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + 2*CU1 + FE1 + MG1 + MN1 - 1.0) - 1.6e-7*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**3 - 3*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2*(-8.0e-8*CR1 - 8.0e-8*CU1 - 8.0e-8*FE1 - 8.0e-8*MG1 - 8.0e-8*MN1 + 8.0e-8) - 9360.0*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 + 107040.0
    return fyy_CU1
@ti.func
def fyy_FCC_A1_FE1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_FE1 = 1.33333333333333*CR1*CU1*(24.982*T - 29976) - 1.33333333333333e-7*CR1*CU1 - 100000.000000007*CR1*MN1 - 2820.0*CR1 + 8108.22*CU1*FE1 + 79999.9999999866*CU1*MG1 + 1.33333333333333*CU1*MN1*(89*T - 121140) - 0.666666666666666*CU1*MN1*(105*T - 60000) - 0.666666666666666*CU1*MN1*(112*T - 160467) - 23333.3333333334*CU1*MN1 - 2.0*CU1*(12687.16 - 8.01*T) + 2.0*CU1*(38590 - 2*T) + 8108.22*CU1*(-2*CU1 + 2*FE1) - 1.2e-7*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2340.0*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 2.4e-7*CU1*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 9360.0*CU1*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 127304.0002*FE1*MN1 + 4.0*FE1*(1.3398*T + 21167.4) + 2.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF2ALMG(T) - 8.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5)*LDF2ALMG(T) + 2.0*MG1*LDF1ALMG(T) + 127304.0*MN1*(CR1/3 + CU1/3 + FE1 + MG1/3) + 53506.0*MN1*(CR1/3 + CU1/3 + MG1/3 + MN1) - 127304.0002*MN1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 0.0002*MN1*(-2*CR1/3 - 2*CU1/3 - FE1 - 2*MG1/3 - MN1 + 1.0) + 17082.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/FE1) - 37.3516*T - 2.0*(1.3398*T + 21167.4)*(-CR1 - CU1 - 2*FE1 - MG1 - MN1 + 1.0) - 4.0*(1.3398*T + 21167.4)*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 152132.2
    return fyy_FE1
@ti.func
def fyy_FCC_A1_MG1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_MG1 = -100000.000000007*CR1*MN1 - 2.0e-6*CR1 + 119999.99999998*CU1*MG1 - 23333.3333333334*CU1*MN1 + 2.0*CU1*(38590 - 2*T) - 2.0e-8*CU1*(CR1/3 + CU1 + FE1/3 + MN1/3) - 2.0e-8*CU1*(CR1/3 + FE1/3 + MG1 + MN1/3) - 1.2e-7*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) - 117659.99999998*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 120000.0*CU1*(-2*CR1/3 - CU1 - 2*FE1/3 - MG1 - 2*MN1/3 + 1.0) + 2.4e-7*CU1*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 9360.0*CU1*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 60270.0001333334*FE1*MN1 + 2.0*FE1*(1.3398*T + 21167.4) + 8.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF2ALMG(T) - 8.0*MG1*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0)*LDF2ALMG(T) + 4.0*MG1*LDF1ALMG(T) + 17600.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/MG1) + (-4.0*CR1 - 4.0*CU1 - 4.0*FE1 - 4.0*MG1 - 4.0*MN1 + 4.0)*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0)*LDF2ALMG(T) - (-1.0*CR1 - 1.0*CU1 - 1.0*FE1 - 2.0*MG1 - 1.0*MN1 + 1.0)*LDF1ALMG(T) - 1.0*(-CR1 - CU1 - FE1 - 2*MG1 - MN1 + 1.0)*LDF1ALMG(T) + 4.0*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1 + CU1 + FE1 + 2*MG1 + MN1 - 1.0)*LDF2ALMG(T) - 4.0*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF1ALMG(T) - 8.0*(-CR1/2 - CU1/2 - FE1/2 - MG1 - MN1/2 + 0.5)**2*LDF2ALMG(T) - 2.0*LDF0ALMG(T)
    return fyy_MG1
@ti.func
def fyy_FCC_A1_MN1(T:float,CR1:float,CU1:float,FE1:float,MG1:float,MN1:float) -> float:
    fyy_MN1 = -100000.00000002*CR1*MN1 - 100000.0*CR1*(CR1 + CU1/3 + FE1/3 + MG1/3) - 2.0e-8*CR1*(CU1/3 + FE1/3 + MG1/3 + MN1) + 100000.00000002*CR1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 100000.0*CR1*(-CR1 - 2*CU1/3 - 2*FE1/3 - 2*MG1/3 - MN1 + 1.0) + 52800.0*CR1 - 0.666666666666666*CU1*FE1*(89*T - 121140) - 0.666666666666666*CU1*FE1*(105*T - 60000) + 1.33333333333333*CU1*FE1*(112*T - 160467) + 79999.9999999866*CU1*MG1 + 2075.12*CU1*MN1 + 2.0*CU1*(38590 - 2*T) + 2075.12*CU1*(-2*CU1 + 2*MN1) - 2.0*CU1*(2.9399*T - 12154.853) - 100000.0*CU1*(CR1/3 + CU1 + FE1/3 + MG1/3) - 30000.0*CU1*(CR1/3 + FE1/3 + MG1/3 + MN1) - 1.2e-7*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 2340.0*CU1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 30000.0*CU1*(-2*CR1/3 - CU1 - 2*FE1/3 - 2*MG1/3 - MN1 + 1.0) + 2.4e-7*CU1*(-CR1/2 - CU1 - FE1/2 - MG1/2 - MN1/2 + 0.5)**2 - 9360.0*CU1*(CR1/2 + CU1 + FE1/2 + MG1/2 + MN1/2 - 0.5) + 52800.0*CU1 + 53506.0002*FE1*MN1 + 2.0*FE1*(1.3398*T + 21167.4) + 127304.0*FE1*(CR1/3 + CU1/3 + FE1 + MG1/3) + 53506.0*FE1*(CR1/3 + CU1/3 + MG1/3 + MN1) - 53506.0002*FE1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) - 0.0002*FE1*(-2*CR1/3 - 2*CU1/3 - FE1 - 2*MG1/3 - MN1 + 1.0) + 53318.0*FE1 + 2.0*MG1*(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0)*LDF2ALMG(T) - 8.0*MG1*(CR1/2 + CU1/2 + FE1/2 + MG1 + MN1/2 - 0.5)*LDF2ALMG(T) + 2.0*MG1*LDF1ALMG(T) + 52800.0*MG1 + 105600.0*MN1 + 8.3145*T*(1.0/(-CR1 - CU1 - FE1 - MG1 - MN1 + 1.0) + 1.0/MN1) - 50.0*T + 85800.0
    return fyy_MN1

@ti.func
def cal_f(i:int,T:float,y:float) -> float:
    f = 0.0
    if   i == 0: f = f_LIQUID(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
    elif i == 1: f = f_FCC_A1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
    return f

@ti.func
def cal_fyy(a:int,j:int,T:float,y:float) -> float:
    fyy = 0.0
    if   a == 0: # if sublattice 1 in LIQUID
        if   j == 0: fyy = fyy_LIQUID_CR1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 1: fyy = fyy_LIQUID_CU1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 2: fyy = fyy_LIQUID_FE1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 3: fyy = fyy_LIQUID_MG1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 4: fyy = fyy_LIQUID_MN1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
    elif a == 1: # if sublattice 1 in FCC_A1
        if   j == 0: fyy = fyy_FCC_A1_CR1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
        elif j == 1: fyy = fyy_FCC_A1_CU1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
        elif j == 2: fyy = fyy_FCC_A1_FE1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
        elif j == 3: fyy = fyy_FCC_A1_MG1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
        elif j == 4: fyy = fyy_FCC_A1_MN1(T,y[1,0],y[1,1],y[1,2],y[1,3],y[1,4])
    return fyy

