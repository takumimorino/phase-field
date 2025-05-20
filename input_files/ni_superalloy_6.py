import numpy as np
import taichi as ti
 
# solutes     = ['AL', 'CO', 'CR', 'FE', 'HF'] #溶質
# uini_ave    = np.array([0.15, 0.02, 0.02, 0.02, 0.015])
# yini_FCC_A1 = np.array([[0.11826812103607044, 0.02927951246610502, 0.02696429568394365, 0.02294299403251599, 0.0018198960236890458]])
# yini_L12 = np.array([[0.7085459911740871, 0.026920815929454823, 0.04447745967613836, 0.0666221943191848, 0.10537921824906257],
#                    [0.00023576135914717337, 0.007043233664899871, 0.0038481263297955553, 0.0010816795490387555, 1.7512093589266284e-08]])

solutes     = ['AL', 'CO', 'CR', 'FE', 'HF']
uini_ave    = np.array([0.169, 0.05, 0.05, 0.01, 0.003])
yini_FCC_A1 = np.array([[0.13475680301264484, 0.07140849313691613, 0.06946883161331197, 0.011114225595998173, 0.0004186765071673434]])
yini_L12 = np.array([[0.8087115284329298, 0.03924113942701601, 0.06737598864227617, 0.03321052049727175, 0.022320219687882798],
                   [0.0013981141331633326, 0.025055574695888202, 0.018262243479164757, 0.0007782515018781279, 9.97309367915747e-09]])


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


@ti.func
def f_FCC_A1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    NI1 = (1.0-AL1-CO1-CR1-FE1-HF1)
    return 1/1.0*(+ \
           + 8.3145*T*( +1.0*(+(AL1)*ti.log(AL1) +(CO1)*ti.log(CO1) +(CR1)*ti.log(CR1) +(FE1)*ti.log(FE1) +(HF1)*ti.log(HF1) +(NI1)*ti.log(NI1) ) )+ \
           + AL1                                       *(+GHSERAL(T))+ \
           + CO1                                       *(+GCOFCC(T))+ \
           + CR1                                       *(+7284+0.163*T+GHSERCR(T))+ \
           + FE1                                       *(-1462.4+8.282*T-1.15*T*ti.log(T)+0.00064*T**2+GHSERFE(T))+ \
           + HF1                                       *(+GHFFCC(T))+ \
           + NI1                                       *(+GHSERNI(T))+ \
           + AL1*CO1                                   *(-105000)+ \
           + AL1*CR1                                   *(-45900+6*T)+ \
           + AL1*FE1                                   *(-76066.1+18.6758*T)+ \
           + AL1*FE1*(AL1-FE1)                         *(+21167.4+1.3398*T)+ \
           + AL1*HF1                                   *(-111979)+ \
           + AL1*NI1                                   *(-162407.75+16.212965*T)+ \
           + AL1*NI1*(AL1-NI1)                         *(+73417.798-34.914168*T)+ \
           + AL1*NI1*(AL1-NI1)**2                      *(+33471.014-9.8373558*T)+ \
           + AL1*NI1*(AL1-NI1)**3                      *(-30758.01+10.25267*T)+ \
           + CO1*CR1                                   *(+1500-9.592*T)+ \
           + CO1*FE1                                   *(-8968.75)+ \
           + CO1*FE1*(CO1-FE1)**2                      *(+3528.8)+ \
           + CO1*HF1                                   *(-83611)+ \
           + CO1*NI1                                   *(-800+1.2629*T)+ \
           + CR1*FE1                                   *(+10833-7.477*T)+ \
           + CR1*FE1*(CR1-FE1)                         *(+1410)+ \
           + CR1*HF1                                   *(+1E-8)+ \
           + CR1*NI1                                   *(+8030-12.8801*T)+ \
           + CR1*NI1*(CR1-NI1)                         *(+33080-16.0362*T)+ \
           + FE1*HF1                                   *(-17000-3.5*T)+ \
           + FE1*NI1                                   *(-12054.355+3.27413*T)+ \
           + FE1*NI1*(FE1-NI1)                         *(+11082.1315-4.45077*T)+ \
           + FE1*NI1*(FE1-NI1)**2                      *(-725.805174)+ \
           + HF1*NI1                                   *(-118000)+ \
           + HF1*NI1*(HF1-NI1)                         *(+1E-8)+ \
           + AL1*CO1*NI1                               *(-7500+5*T)+ \
           + AL1*CR1*NI1                               *(+30300)+ \
           + AL1*FE1*NI1*(AL1+(1/3)*(1-(AL1+FE1+NI1))) *(+1E-8)+ \
           + AL1*FE1*NI1*(FE1+(1/3)*(1-(AL1+FE1+NI1))) *(-77900)+ \
           + AL1*FE1*NI1*(NI1+(1/3)*(1-(AL1+FE1+NI1))) *(+58000)+ \
           + AL1*HF1*NI1                               *(-90000)+ \
           + CO1*CR1*NI1                               *(-20000+10*T)+ \
           + CR1*FE1*NI1                               *(+1618) )

@ti.func
def fyy_FCC_A1_AL1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    fyy_AL1 = 115999.99999998*AL1*FE1 + 8.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(2*AL1 + CO1 + CR1 + FE1 + HF1 - 1.0) - 4.0*AL1*(73417.798 - 34.914168*T) + 24.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(2*AL1 + CO1 + CR1 + FE1 + HF1 - 1.0) - 48.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 - 2.0*CO1*(5*T - 7500) - 2.0*CR1*(33080 - 16.0362*T) - 60600.0*CR1 - 2.0*FE1*(11082.1315 - 4.45077*T) + 2.0*FE1*(1.3398*T + 21167.4) - 2.0e-8*FE1*(AL1 + CO1/3 + CR1/3 + HF1/3) + 155800.0*FE1*(CO1/3 + CR1/3 + FE1 + HF1/3) - 117451.61034798*FE1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 116000.0*FE1*(-AL1 - 2*CO1/3 - 2*CR1/3 - FE1 - 2*HF1/3 + 1.0) + 5806.441392*FE1*(AL1/2 + CO1/2 + CR1/2 + FE1 + HF1/2 - 0.5) + 179999.99999998*HF1 + 8.3145*T*(1.0/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 1.0/AL1) - 32.42593*T + 4.0*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(2*AL1 + CO1 + CR1 + FE1 + HF1 - 1.0) - 4.0*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 + 4.0*(73417.798 - 34.914168*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 2.0*(73417.798 - 34.914168*T)*(2*AL1 + CO1 + CR1 + FE1 + HF1 - 1.0) + (133884.056 - 39.3494232*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(2*AL1 + CO1 + CR1 + FE1 + HF1 - 1.0) - (133884.056 - 39.3494232*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 + 24.0*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 - 8.0*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**3 + 3*(82.02136*T - 246064.08)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 - (82.02136*T - 246064.08)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**3 + 324815.5
    return fyy_AL1
@ti.func
def fyy_FCC_A1_CO1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    fyy_CO1 = 129266.66666666*AL1*FE1 + 2.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 2.0*AL1*(73417.798 - 34.914168*T) - 2.0*AL1*(5*T - 7500) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 24.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 + 7057.6*CO1*FE1 - 2.0*CR1*(33080 - 16.0362*T) - 2.0*CR1*(10*T - 20000) - 2.0*FE1*(11082.1315 - 4.45077*T) + 7057.6*FE1*(2*CO1 - 2*FE1) - 1451.610348*FE1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 5806.441392*FE1*(AL1/2 + CO1/2 + CR1/2 + FE1 + HF1/2 - 0.5) - 2.0e-8*HF1 + 8.3145*T*(1.0/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 1.0/CO1) - 2.5258*T + 1600.0
    return fyy_CO1
@ti.func
def fyy_FCC_A1_CR1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    fyy_CR1 = 129266.66666666*AL1*FE1 + 2.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 2.0*AL1*(73417.798 - 34.914168*T) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 24.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 - 60600.0*AL1 - 2.0*CO1*(10*T - 20000) - 4.0*CR1*(33080 - 16.0362*T) - 2.0*FE1*(11082.1315 - 4.45077*T) - 1451.610348*FE1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 5806.441392*FE1*(AL1/2 + CO1/2 + CR1/2 + FE1 + HF1/2 - 0.5) - 416.0*FE1 - 2.0e-8*HF1 + 8.3145*T*(1.0/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 1.0/CR1) + 25.7602*T + 2*(33080.0 - 16.0362*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - (33080.0 - 16.0362*T)*(AL1 + CO1 + 2*CR1 + FE1 + HF1 - 1.0) + 2.0*(33080 - 16.0362*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 1.0*(33080 - 16.0362*T)*(AL1 + CO1 + 2*CR1 + FE1 + HF1 - 1.0) - 16060.0
    return fyy_CR1
@ti.func
def fyy_FCC_A1_FE1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    fyy_FE1 = 271800.0*AL1*FE1 + 2.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 2.0*AL1*(73417.798 - 34.914168*T) - 2.0*AL1*(1.3398*T + 21167.4) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 24.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 - 2.0e-8*AL1*(AL1 + CO1/3 + CR1/3 + HF1/3) + 155800.0*AL1*(CO1/3 + CR1/3 + FE1 + HF1/3) - 271800.0*AL1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 116000.0*AL1*(-AL1 - 2*CO1/3 - 2*CR1/3 - FE1 - 2*HF1/3 + 1.0) + 7057.6*CO1*FE1 + 7057.6*CO1*(-2*CO1 + 2*FE1) - 2.0*CR1*(33080 - 16.0362*T) - 6056.0*CR1 - 4.0*FE1*(11082.1315 - 4.45077*T) - 5806.441392*FE1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 5806.441392*FE1*(AL1 + CO1 + CR1 + 2*FE1 + HF1 - 1.0) - 2.0e-8*HF1 + 8.3145*T*(1.0/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 1.0/FE1) - 6.54826*T + 4.0*(11082.1315 - 4.45077*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 2.0*(11082.1315 - 4.45077*T)*(AL1 + CO1 + CR1 + 2*FE1 + HF1 - 1.0) - (-2903.220696*AL1 - 2903.220696*CO1 - 2903.220696*CR1 - 2903.220696*FE1 - 2903.220696*HF1 + 2903.220696)*(AL1 + CO1 + CR1 + 2*FE1 + HF1 - 1.0) - 2903.220696*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1 + CR1 + 2*FE1 + HF1 - 1.0) + 5806.441392*(AL1/2 + CO1/2 + CR1/2 + FE1 + HF1/2 - 0.5)**2 + 24108.71
    return fyy_FE1
@ti.func
def fyy_FCC_A1_HF1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float) -> float:
    fyy_HF1 = 129266.66666666*AL1*FE1 + 2.0*AL1*(33471.014 - 9.8373558*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 8.0*AL1*(33471.014 - 9.8373558*T)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 2.0*AL1*(73417.798 - 34.914168*T) + 12.0*AL1*(10.25267*T - 30758.01)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5) - 24.0*AL1*(10.25267*T - 30758.01)*(AL1 + CO1/2 + CR1/2 + FE1/2 + HF1/2 - 0.5)**2 + 179999.99999994*AL1 - 6.0e-8*CO1 - 2.0*CR1*(33080 - 16.0362*T) - 6.0e-8*CR1 - 2.0*FE1*(11082.1315 - 4.45077*T) - 1451.610348*FE1*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 5806.441392*FE1*(AL1/2 + CO1/2 + CR1/2 + FE1 + HF1/2 - 0.5) - 6.0e-8*FE1 - 1.2e-7*HF1 + 8.3145*T*(1.0/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 1.0/HF1) + 236000.00000006
    return fyy_HF1

@ti.func
def f_L12(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    NI1 = (1.0-AL1-CO1-CR1-FE1-HF1)
    NI2 = (1.0-AL2-CO2-CR2-FE2-HF2)
    return 1/1.0*(+ \
           + 8.3145*T*( +0.25*(+(AL1)*ti.log(AL1) +(CO1)*ti.log(CO1) +(CR1)*ti.log(CR1) +(FE1)*ti.log(FE1) +(HF1)*ti.log(HF1) +(NI1)*ti.log(NI1) )  +0.75*(+(AL2)*ti.log(AL2) +(CO2)*ti.log(CO2) +(CR2)*ti.log(CR2) +(FE2)*ti.log(FE2) +(HF2)*ti.log(HF2) +(NI2)*ti.log(NI2) ) )+ \
           + AL1                                       *AL2                                       *(+GHSERAL(T)+5000)+ \
           + AL1                                       *CO2                                       *(-21000+0.25*GHSERAL(T)+0.75*GHSERCO(T))+ \
           + AL1                                       *FE2                                       *(+0.25*GHSERAL(T)+0.75*GHSERFE(T))+ \
           + AL1                                       *NI2                                       *(-43000+5.5*T+0.75*GHSERNI(T)+0.25*GHSERAL(T))+ \
           + CO1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERCO(T)-22600)+ \
           + CO1                                       *CO2                                       *(+GHSERCO(T)+15000)+ \
           + CO1                                       *FE2                                       *(+0.75*GHSERFE(T)+0.25*GHSERCO(T))+ \
           + CO1                                       *NI2                                       *(-1600-3*T+0.75*GHSERNI(T)+0.25*GHSERCO(T))+ \
           + CR1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERCR(T))+ \
           + CR1                                       *CO2                                       *(+10000-10*T+0.75*GHSERCO(T)+0.25*GHSERCR(T))+ \
           + CR1                                       *CR2                                       *(+35000+GHSERCR(T))+ \
           + CR1                                       *FE2                                       *(+20000+0.75*GHSERFE(T)+0.25*GHSERCR(T))+ \
           + CR1                                       *NI2                                       *(+1650-5*T+0.75*GHSERNI(T)+0.25*GHSERCR(T))+ \
           + AL1                                       *CR2                                       *(-12000+0.25*GHSERAL(T)+0.75*GHSERCR(T))+ \
           + CO1                                       *CR2                                       *(+0.25*GHSERCO(T)+0.75*GHSERCR(T)+10000-10*T)+ \
           + FE1                                       *CR2                                       *(+1000+0.25*GHSERFE(T)+0.75*GHSERCR(T))+ \
           + NI1                                       *CR2                                       *(+0.25*GHSERNI(T)+0.75*GHSERCR(T))+ \
           + FE1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERFE(T))+ \
           + FE1                                       *CO2                                       *(+0.75*GHSERCO(T)+0.25*GHSERFE(T))+ \
           + FE1                                       *FE2                                       *(+GHSERFE(T)+5000)+ \
           + FE1                                       *NI2                                       *(+0.75*GDHCNI(T)+0.25*GDHCFE(T)+5000)+ \
           + NI1                                       *AL2                                       *(+0.75*GHSERAL(T)+0.25*GHSERNI(T))+ \
           + NI1                                       *CO2                                       *(+0.75*GHSERCO(T)+0.25*GHSERNI(T))+ \
           + NI1                                       *FE2                                       *(+0.75*GDHCFE(T)+0.25*GDHCNI(T)+5000)+ \
           + NI1                                       *NI2                                       *(+GHSERNI(T)+3900)+ \
           + AL1                                       *AL2*NI2                                   *(-100000+15*T)+ \
           + AL1*CR1                                   *NI2                                       *(-5000+5*T)+ \
           + AL1*NI1                                   *NI2                                       *(+2000-2*T)+ \
           + CO1*CR1                                   *NI2                                       *(+2000-3*T)+ \
           + CO1*CR1*(CO1-CR1)                         *NI2                                       *(+1E-8)+ \
           + CO1*CR1*(CO1-CR1)**2                      *NI2                                       *(+1E-8)+ \
           + CR1                                       *CO2*NI2                                   *(+1E-8)+ \
           + CR1                                       *CO2*NI2*(CO2-NI2)                         *(+1E-8)+ \
           + CR1                                       *CO2*NI2*(CO2-NI2)**2                      *(+1E-8)+ \
           + AL1*CO1                                   *NI2                                       *(+5000-3*T)+ \
           + AL1*CO1*(AL1-CO1)                         *NI2                                       *(+2500)+ \
           + AL1*CO1*(AL1-CO1)**2                      *NI2                                       *(+1E-8)+ \
           + AL1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + CO1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + CR1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + FE1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + HF1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + NI1                                       *CR2*NI2                                   *(5000-3*T)+ \
           + HF1                                       *AL2                                       *(+0.25*GHSERHF(T)+0.75*GHSERAL(T))+ \
           + HF1                                       *CO2                                       *(+0.25*GHSERHF(T)+0.75*GHSERCO(T))+ \
           + HF1                                       *CR2                                       *(+0.25*GHSERHF(T)+0.75*GHSERCR(T))+ \
           + HF1                                       *FE2                                       *(+0.25*GHSERHF(T)+0.75*GHSERFE(T))+ \
           + HF1                                       *HF2                                       *(+0.25*GHSERHF(T)+0.75*GHSERHF(T)+10000)+ \
           + HF1                                       *NI2                                       *(+0.75*GHSERNI(T)+0.25*GHSERHF(T)-39500)+ \
           + AL1                                       *HF2                                       *(+0.75*GHSERHF(T)+0.25*GHSERAL(T))+ \
           + CO1                                       *HF2                                       *(+0.75*GHSERHF(T)+0.25*GHSERCO(T))+ \
           + CR1                                       *HF2                                       *(+0.75*GHSERHF(T)+0.25*GHSERCR(T))+ \
           + FE1                                       *HF2                                       *(+0.75*GHSERHF(T)+0.25*GHSERFE(T))+ \
           + NI1                                       *HF2                                       *(+0.25*GHSERNI(T)+0.75*GHSERHF(T)-20000)+ \
           + AL1*HF1                                   *NI2                                       *(+1E-8)+ \
           + AL1*HF1*(AL1-HF1)                         *NI2                                       *(-2500)+ \
           + AL1*HF1*(AL1-HF1)**2                      *NI2                                       *(+1E-8)+ \
           + AL1*FE1                                   *NI2                                       *(-20000+3*T)+ \
           + CR1*FE1                                   *NI2                                       *(-20000+3*T)+ \
           + FE1*HF1                                   *NI2                                       *(-20000+3*T)+ \
           + AL1*CR1                                   *CO2                                       *(-50000+55*T) )

@ti.func
def fyy_L12_AL1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_AL1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*AL1*HF1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*CO1*(2*AL1 - 2*CO1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 5000.0*CO1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*HF1*(2*AL1 - 2*HF1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 5000.0*HF1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 0.25/AL1) - 1.0*(2000 - 2*T)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - (2000.0 - 2.0*T)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0)
    return fyy_AL1
@ti.func
def fyy_L12_CO1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_CO1 = 2.0e-8*AL1*CO1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*AL1*(-2*AL1 + 2*CO1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 5000.0*AL1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*CO1*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*CR1*(2*CO1 - 2*CR1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 0.25/CO1)
    return fyy_CO1
@ti.func
def fyy_L12_CR1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_CR1 = 2.0e-8*CO1*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*CO1*(-2*CO1 + 2*CR1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 2.0e-8*CO1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 0.25/CR1)
    return fyy_CR1
@ti.func
def fyy_L12_FE1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_FE1 = 8.3145*T*(0.25/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 0.25/FE1)
    return fyy_FE1
@ti.func
def fyy_L12_HF1(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_HF1 = 2.0e-8*AL1*HF1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 2.0e-8*AL1*(-2*AL1 + 2*HF1)*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 5000.0*AL1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 8.3145*T*(0.25/(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) + 0.25/HF1)
    return fyy_HF1
@ti.func
def fyy_L12_AL2(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_AL2 = -2.0*AL1*(15*T - 100000) + 2.0e-8*CO2*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CO2*CR1*(AL2/2 + CO2 + CR2/2 + FE2/2 + HF2/2 - 0.5) - 2.0e-8*CO2*CR1 + 8.3145*T*(0.75/(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 0.75/AL2)
    return fyy_AL2
@ti.func
def fyy_L12_CO2(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_CO2 = 8.0e-8*CO2*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CO2*CR1*(AL2 + 2*CO2 + CR2 + FE2 + HF2 - 1.0) - 4.0e-8*CO2*CR1 + 8.0e-8*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0)*(AL2 + 2*CO2 + CR2 + FE2 + HF2 - 1.0) + 4.0e-8*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CR1*(AL2/2 + CO2 + CR2/2 + FE2/2 + HF2/2 - 0.5)**2 - 2.0e-8*CR1*(AL2 + 2*CO2 + CR2 + FE2 + HF2 - 1.0) - 2.0e-8*CR1 + 8.3145*T*(0.75/(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 0.75/CO2)
    return fyy_CO2
@ti.func
def fyy_L12_CR2(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_CR2 = -2.0*AL1*(5000 - 3*T) - 2.0*CO1*(5000 - 3*T) + 2.0e-8*CO2*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CO2*CR1*(AL2/2 + CO2 + CR2/2 + FE2/2 + HF2/2 - 0.5) - 2.0e-8*CO2*CR1 - 2.0*CR1*(5000 - 3*T) - 2.0*FE1*(5000 - 3*T) - 2.0*HF1*(5000 - 3*T) + 8.3145*T*(0.75/(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 0.75/CR2) - (5000.0 - 3.0*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0) - 1.0*(5000 - 3*T)*(-AL1 - CO1 - CR1 - FE1 - HF1 + 1.0)
    return fyy_CR2
@ti.func
def fyy_L12_FE2(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_FE2 = 2.0e-8*CO2*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CO2*CR1*(AL2/2 + CO2 + CR2/2 + FE2/2 + HF2/2 - 0.5) - 2.0e-8*CO2*CR1 + 8.3145*T*(0.75/(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 0.75/FE2)
    return fyy_FE2
@ti.func
def fyy_L12_HF2(T:float,AL1:float,CO1:float,CR1:float,FE1:float,HF1:float,AL2:float,CO2:float,CR2:float,FE2:float,HF2:float) -> float:
    fyy_HF2 = 2.0e-8*CO2*CR1*(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) - 8.0e-8*CO2*CR1*(AL2/2 + CO2 + CR2/2 + FE2/2 + HF2/2 - 0.5) - 2.0e-8*CO2*CR1 + 8.3145*T*(0.75/(-AL2 - CO2 - CR2 - FE2 - HF2 + 1.0) + 0.75/HF2)
    return fyy_HF2

@ti.func
def cal_f(i:int,T:float,y:float) -> float:
    f = 0.0
    if   i == 0: 
        f = f_FCC_A1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
    elif i == 1  or i == 2 or i == 3 or i == 4:
        a = i*2 - 1 
        f = f_L12(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
    return f

@ti.func
def cal_fyy(a:int,j:int,T:float,y:float) -> float:
    fyy = 0.0
    if   a == 0:
        if   j == 0: fyy = fyy_FCC_A1_AL1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 1: fyy = fyy_FCC_A1_CO1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 2: fyy = fyy_FCC_A1_CR1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 3: fyy = fyy_FCC_A1_FE1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
        elif j == 4: fyy = fyy_FCC_A1_HF1(T,y[0,0],y[0,1],y[0,2],y[0,3],y[0,4])
    elif a == 1 or a == 3 or a == 5 or a == 7:
        if   j == 0: fyy = fyy_L12_AL1(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
        elif j == 1: fyy = fyy_L12_CO1(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
        elif j == 2: fyy = fyy_L12_CR1(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
        elif j == 3: fyy = fyy_L12_FE1(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
        elif j == 4: fyy = fyy_L12_HF1(T,y[a,0],y[a,1],y[a,2],y[a,3],y[a,4],y[a+1,0],y[a+1,1],y[a+1,2],y[a+1,3],y[a+1,4])
    elif a == 2 or a == 4 or a == 6 or a == 8:
        if   j == 0: fyy = fyy_L12_AL2(T,y[a-1,0],y[a-1,1],y[a-1,2],y[a-1,3],y[a-1,4],y[a,0],y[a,1],y[a,2],y[a,3],y[a,4])
        elif j == 1: fyy = fyy_L12_CO2(T,y[a-1,0],y[a-1,1],y[a-1,2],y[a-1,3],y[a-1,4],y[a,0],y[a,1],y[a,2],y[a,3],y[a,4])
        elif j == 2: fyy = fyy_L12_CR2(T,y[a-1,0],y[a-1,1],y[a-1,2],y[a-1,3],y[a-1,4],y[a,0],y[a,1],y[a,2],y[a,3],y[a,4])
        elif j == 3: fyy = fyy_L12_FE2(T,y[a-1,0],y[a-1,1],y[a-1,2],y[a-1,3],y[a-1,4],y[a,0],y[a,1],y[a,2],y[a,3],y[a,4])
        elif j == 4: fyy = fyy_L12_HF2(T,y[a-1,0],y[a-1,1],y[a-1,2],y[a-1,3],y[a-1,4],y[a,0],y[a,1],y[a,2],y[a,3],y[a,4])
    return fyy

