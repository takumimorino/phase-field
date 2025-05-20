import numpy as np
import time
from matplotlib import pyplot as plt
import taichi as ti
import os

if 'superalloy' in os.getenv("import_file"): from input_files.ni_superalloy_common import *
elif 'dendrite' in os.getenv("import_file"): from input_files.dendrite_common import *
if 1 == 2:                                   from input_files.ni_dendrite_3 import * # dummy


# ----------------------------↓ Variables & Constants ↓---------------------------------------------
number_of_solutes     = len(solutes)
number_of_phases      = len(phases)
number_of_sublattices = int(np.sum(set_y_num))
pi                    = np.pi
dimension             = sum(1 for v in (nx, ny, nz) if v > 1)
dt_max                = dx**2/((dimension*2.0 + 1.0)*np.max(set_D)) # Courant condition


nstep                 = ti.field(int,   shape=())
solidification_time   = ti.field(float, shape=())
dt                    = ti.field(float, shape=())                                         ; dt.fill(dt_max)
y_num                 = ti.field(float, shape=(len(set_y_num)))                           ; y_num.from_numpy(set_y_num)
y_S                   = ti.field(float, shape=(len(set_y_S)))                             ; y_S.from_numpy(set_y_S)
y_phase_ID            = ti.field(int,   shape=(len(set_y_phase_ID)))                      ; y_phase_ID.from_numpy(set_y_phase_ID.astype(np.int32))
aij                   = ti.field(float, shape=(number_of_phases,number_of_phases))        ; aij.from_numpy(2.0 / pi * np.sqrt(2.0*delta*set_sigma))
wij                   = ti.field(float, shape=(number_of_phases,number_of_phases))        ; wij.from_numpy(4.0 * set_sigma/delta)
sigma                 = ti.field(float, shape=(number_of_phases, number_of_phases ))      ; sigma.from_numpy(set_sigma)
Vm                    = ti.field(float, shape=(number_of_phases))                         ; Vm.from_numpy(set_Vm)
yini                  = ti.field(float, shape=(number_of_sublattices,number_of_solutes))  ; yini.from_numpy(set_yini)
D                     = ti.field(float, shape=(number_of_phases,number_of_solutes))       ; D.from_numpy(set_D)


y                     = ti.Matrix.field(n=number_of_sublattices, m=number_of_solutes, dtype=float, shape=(nx,ny,nz), needs_grad=True) # site fraction
y_new                 = ti.Matrix.field(n=number_of_sublattices, m=number_of_solutes, dtype=float, shape=(nx,ny,nz))                  # site fraction at time t
ui                    = ti.Matrix.field(n=number_of_phases, m=number_of_solutes, dtype=float, shape=(nx,ny,nz))                       # phase composition
u                     = ti.Vector.field(number_of_solutes, dtype=float, shape=(nx,ny,nz))                                             # composition
f                     = ti.Vector.field(number_of_phases, dtype=float, shape=(nx,ny,nz), needs_grad=True)                             # chemical free energy
fyy                   = ti.Matrix.field(n=number_of_sublattices, m=number_of_solutes, dtype=float, shape=(nx,ny,nz))                  # second derivative of the chemical free energy
phi                   = ti.Vector.field(number_of_phases,  dtype=float, shape=(nx,ny,nz))                                             # phase-field variable at time t
phi_new               = ti.Vector.field(number_of_phases,  dtype=float, shape=(nx,ny,nz))                                             # phase-field variable at time t+dt
dpdt                  = ti.Vector.field(number_of_phases,  dtype=float, shape=(nx,ny,nz))                                             # time derivative of the phase-field variable
dudt                  = ti.Vector.field(number_of_solutes, dtype=float, shape=(nx,ny,nz))                                             # time drivative of the composition
T                     = ti.field(dtype=float, shape=(nx,ny,nz)); T.fill(Tini)                                                         # Temperature
nf                    = ti.field(dtype=int, shape=(nx,ny,nz))                                                                         # number of phases at each computational grid (Include the neighboring pixels as well)
nfe                   = ti.field(dtype=int, shape=(nx,ny,nz))                                                                         # number of phases at each computational grid (Exclude the neighboring pixels)
mf                    = ti.Vector.field(number_of_phases, dtype=int, shape=(nx,ny,nz))                                                # phase IDs at each computational grid (Include the neighboring pixels as well)
mfe                   = ti.Vector.field(number_of_phases, dtype=int, shape=(nx,ny,nz))                                                # phase IDs at each computational grid (Exclude the neighboring pixels)
eij                   = ti.Matrix.field(n=number_of_phases, m=number_of_phases , dtype=float, shape=(nx,ny,nz))                       # driving force for interface movement due to the chemical free energy
mij                   = ti.Matrix.field(n=number_of_phases, m=number_of_phases , dtype=float, shape=(nx,ny,nz))                       # phase-field mobility
set_PF_mobility       = ( set_interface_mobility * pi**2/(4*delta) ).tolist()
@ti.kernel
def fill_mij():
    for ID in ti.grouped(mij):
        mij[ID] = set_PF_mobility
fill_mij()


eij_elast             = np.zeros((nx,ny,nz))                    # driving force for interface movement due to the elastic free energy (ndarray)
eij_elast_taichi      = ti.field(dtype=float, shape=(nx,ny,nz)) # driving force for interface movement due to the elastic free energy (taichi field)
homo_strain           = np.zeros((nx,ny))                       # homogeneous strain
inhomo_strain         = np.zeros((3,nx,ny))                     # inhomogeneous strain
elastic_strain        = np.zeros((3,nx,ny))                     # elastic starin
stress                = np.zeros((3,nx,ny))                     # stress
eigen_strain_inv      = np.empty((nx,ny), dtype=np.complex64)   # eigen strain (reciprocal space)
homo_strain_inv       = np.empty((3,nx,ny), dtype=np.complex64) # homogeneous strain (reciprocal space)
inhomo_strain_inv     = np.empty((3,nx,ny), dtype=np.complex64) # inhomogeneous strain (reciprocal space)


# Functions for calculating wave number vectors k and k^2
def calc_wave_vector(nx, ny, dx, dy):
    half_nx = int(nx/2)
    half_ny = int(ny/2)
    dkx     = (2.0 * pi) / (nx * dx)
    dky     = (2.0 * pi) / (ny * dy)
    k2      = np.zeros([nx, ny])
    kx_arr  = np.zeros([nx, ny]) 
    ky_arr  = np.zeros([nx, ny]) 
    kxx     = np.zeros([nx, ny]) 
    kyy     = np.zeros([nx, ny]) 
    kxy     = np.zeros([nx, ny]) 
    
    for i in range(nx):
      if i < half_nx:
        kx = i*dkx
      else:
        kx = (i-nx)*dkx
      kx2 = kx**2

      for j in range(ny):
        if j < half_ny:
          ky = j*dky
        else:
          ky = (j-ny)*dky
        ky2 = ky**2

        k2[i,j] = kx2 + ky2

        nnn = np.sqrt(kx**2 + ky**2) # Magnitude of the wavenumber vector
        if nnn == 0.0:
           nnn = 1.0 #　Avoid division by zeros
        kx_arr[i,j] = kx/nnn
        ky_arr[i,j] = ky/nnn
        kxx[i,j] = (kx/nnn)**2
        kyy[i,j] = (ky/nnn)**2
        kxy[i,j] = (kx/nnn)*(ky/nnn)
    return k2, kxx, kyy, kxy, kx_arr, ky_arr

k2, kxx, kyy, kxy, kx, ky = calc_wave_vector(nx, ny, dx, dy)
# ----------------------------↑ Variables & Constants ↑---------------------------------------------



# ----------------------------↓Definition of functions↓---------------------------------------------
@ti.func
def neighbor_index(l,m,k):
    l_p = l + 1
    l_m = l - 1
    m_p = m + 1
    m_m = m - 1
    k_p = k + 1
    k_m = k - 1

    if boundary_condition_x == 'zero_Neumann': # x direction：zeroneumann condition
        if l_p > nx-1 : l_p = nx-1
        if l_m < 0  : l_m = 0
    elif boundary_condition_x == 'periodic': # x direction：periodic condition
        if l_p > nx-1 : l_p = l_p - nx
        if l_m < 0  : l_m = l_m + nx

    if boundary_condition_y == 'zero_Neumann': # y direction：zeroneumann condition
        if m_p > ny-1 : m_p = ny-1
        if m_m < 0  : m_m = 0
    elif boundary_condition_y == 'periodic': # # y direction：periodic condition
        if m_p > ny-1 : m_p = m_p - ny
        if m_m < 0  : m_m = m_m + ny

    if boundary_condition_z == 'zero_Neumann': # z direction：zeroneumann condition
        if k_p > nz-1 : k_p = nz-1
        if k_m < 0  : k_m = 0
    elif boundary_condition_z == 'periodic': # z direction：periodic condition
        if k_p > nz-1 : k_p = k_p - nz
        if k_m < 0  : k_m = k_m + nz
    
    return l_p, l_m, m_p, m_m, k_p, k_m


@ti.kernel
def update_nfmf():
    for l,m,k in phi:
        l_p, l_m, m_p, m_m, k_p, k_m = neighbor_index(l,m,k)

        n = 0
        ne = 0
        
        for i in range(number_of_phases):
            if phi[l,m,k][i] > 0.0 or (phi[l,m,k][i] == 0.0 and phi[l_p,m,k][i] > 0.0 or phi[l_m,m,k][i] > 0.0 or phi[l,m_p,k][i] > 0.0 or phi[l,m_m,k][i] > 0.0 or phi[l,m,k_p][i] > 0.0 or phi[l,m,k_m][i] > 0.0): # Include the neighboring pixels as well
                mf[l,m,k][n] = i
                n += 1
            if phi[l,m,k][i] > 0.0: # Exclude the neighboring pixels as well
                mfe[l,m,k][ne] = i
                ne += 1

        nf[l,m,k] = n
        nfe[l,m,k] = ne


@ti.kernel
def interp_y(): # If the interface comes up to the neighbouring pixel, the site fraction is calculated as a weighted sum of the neighbouring pixels
    for ID in ti.grouped(y): # Loop all pixcels
        y_new[ID] = y[ID]

    for l,m,k in phi: # Loop all pixcels
        for i in range(nf[l,m,k]): # Loop all phases existing at the pixcel [l,m,k]
            I = mf[l,m,k][i]
            
            if phi[l,m,k][I] == 0:
                l_p, l_m, m_p, m_m, k_p, k_m = neighbor_index(l,m,k)
                phi_sum = phi[l_p,m,k][I] + phi[l_m,m,k][I] + phi[l,m_p,k][I] + phi[l,m_m,k][I] + phi[l,m,k_p][I] + phi[l,m,k_m][I] 

                for a in range(y_phase_ID[I], y_phase_ID[I+1]): # Loop all sublattices in phase I
                    for j in range(number_of_solutes):

                        y_sum = y_new[l_p,m  ,k  ][a,j]*phi[l_p,m  ,k  ][I] + \
                                y_new[l_m,m  ,k  ][a,j]*phi[l_m,m  ,k  ][I] + \
                                y_new[l  ,m_p,k  ][a,j]*phi[l  ,m_p,k  ][I] + \
                                y_new[l  ,m_m,k  ][a,j]*phi[l  ,m_m,k  ][I] + \
                                y_new[l  ,m  ,k_p][a,j]*phi[l  ,m  ,k_p][I] + \
                                y_new[l  ,m  ,k_m][a,j]*phi[l  ,m  ,k_m][I]

                        y[l,m,k][a,j] = y_sum/phi_sum


@ti.kernel
def update_ui(): # Phase composition is calculated by summing site fractions
    ui.fill(0.0)
    for ID in ti.grouped(nf): # Loop all pixcels
        for i in range(nf[ID]): # Loop all phases existing at the pixcel [l,m,k]
            I = mf[ID][i]
            for a in range(y_phase_ID[I], y_phase_ID[I+1]): # Loop all sublattices in phase I
                for j in range(number_of_solutes):
                    ui[ID][I,j] += y_S[a]*y[ID][a,j]


@ti.kernel
def update_f(): # Chemical free energy is calculated based on temperature and site fractions # This function can be used to calculate first derivative of the chemical free energy by the differenciable programing of Taichi 
    for ID in ti.grouped(nfe):   
        if nfe[ID] > 1 or y_num[mfe[ID][0]]>1: # Calculate only at the interface or sublattice phase
            for i in range(nfe[ID]):
                I = mfe[ID][i]
                f[ID][I] = cal_f(I, T[ID], y[ID])


@ti.kernel
def update_fyy(): # Second derivative of the chemical free energy is calculated based on temperature and site fractions
    fyy.fill(0.0)
    for ID in ti.grouped(nf):   
        if nf[ID] > 1 or int(y_num[mf[ID][0]])>1: # Calculate only at the interface or sublattice phase
            for i in range(nf[ID]):
                I = mf[ID][i]
                for a in range(y_phase_ID[I], y_phase_ID[I+1]): 
                    for j in range(number_of_solutes):
                        fyy[ID][a,j] = cal_fyy(a, j, T[ID], y[ID])



@ti.kernel
def update_eij(): # Driving force of the interface movement due to chemical free energy is calculated based on ui(phase composition), y.grad(first derivative of the chemical free energy) and f (the chemical free energy)

    eij.fill(0.0)

    for ID in ti.grouped(phi):

        if nfe[ID]>1: # Calculate only at the interface
            
            for i in range(nfe[ID]):
                I = mfe[ID][i]
                f_I  =  f[ID][I]/Vm[I]  # scalar

                for ii in range(i+1, nfe[ID]):
                    II = mfe[ID][ii]

                    f_II =  f[ID][II]/Vm[II] # scalar
                    fu_I = y.grad[ID]/Vm[I]/y_S[I] # vector
                    driving_force = -(f_I-f_II)
                    a_I = y_phase_ID[I] # Sublattice ID of the first sublattice of phase I
                    for j in range(number_of_solutes):
                        driving_force += (ui[ID][I,j]-ui[ID][II,j])*fu_I[a_I,j]
                    eij[ID][I,II] = driving_force


@ti.kernel
def update_mij(): # Phase-field mobility is calculated based on ui(phase composition) and fyy(second derivative of the chemical free energy)
    mij.fill(1e-20) # If set to 0, the interface does not move
    for ID in ti.grouped(mij):
        if nfe[ID]>1:
            for i in range(nfe[ID]):
                p = mfe[ID][i]

                for ii in range(i+1, nfe[ID]):
                    q = mfe[ID][ii]

                    xi = 0.0
                    for j in range(number_of_solutes):
                        fuu_p = fyy[ID][p,j]/y_S[p]**2
                        xi += (ui[ID][p,j]-ui[ID][q,j])**2  * fuu_p / Vm[p] / D[p,j]

                    mobility = 8*sigma[p,q]/pi*ti.sqrt(2*wij[p,q])/((aij[p,q])**3) / xi

                    mij[ID][p,q] = mobility



@ti.kernel
def update_dpdt(): # The multiphase-field equation
    dpdt.fill(0.0)
    # phi_new.fill(0.0) #mfに含まれていない相は計算が飛ばされるから初期化する必要がある（前ステップでは存在していたが、今回ステップで消失した場合にバグが生じる
    
    for l,m,k in dpdt:
        l_p, l_m, m_p, m_m, k_p, k_m = neighbor_index(l,m,k)

        for n1 in range(nf[l,m,k]):
            i = mf[l,m,k][n1]
            
            for n2 in range(n1+1, nf[l,m,k]):
                j = mf[l,m,k][n2]
                ppp = 0.0

                phii_phij = phi[l,m,k][i]*phi[l,m,k][j]
                chem_driv = 8./pi*ti.sqrt(phii_phij)*eij[l,m,k][i,j] 
                elas_driv = 8./pi*ti.sqrt(phii_phij)*eij_elast_taichi[l,m,k]  
                
                sigma_factor = 1.0
                if anisotoropy: # Calculation of the anisotoropy
                    xi = -15*nu
                    dpdx = (phi[l_p,m  ,k  ][j] - phi[l_m,m  ,k  ][j])/(2*dx)
                    dpdy = (phi[l  ,m_p,k  ][j] - phi[l  ,m_m,k  ][j])/(2*dy)
                    dpdz = (phi[l  ,m  ,k_p][j] - phi[l  ,m  ,k_m][j])/(2*dz)
                    norm_4 = (dpdx**4 + dpdy**4 + dpdz**4) / ((dpdx**2 + dpdy**2 + dpdz**2)**2 + 1e-10) # 4乗の項
                    sigma_factor = (1-3.0*xi) * (1 + 4*xi/(1-3.0*xi)*norm_4)
                
                for n3 in range(nf[l,m,k]):
                    h = mf[l,m,k][n3]
                    laplacian   =  (phi[l_p,m,k][h] + \
                                    phi[l_m,m,k][h] + \
                                    phi[l,m_p,k][h] + \
                                    phi[l,m_m,k][h] + \
                                    phi[l,m,k_p][h] + \
                                    phi[l,m,k_m][h] - 6.0*phi[l,m,k][h])/dx/dx

                    ppp += sigma_factor * ((wij[i,h]-wij[j,h])*phi[l,m,k][h] + 0.5*(aij[i,h]**2 - aij[j,h]**2)*laplacian) #（Lecture on Phase-Field P37 Eq. (3.17))

                dpi = - 2.0 * mij[l,m,k][i,j] / float(nf[l,m,k]) * (ppp - chem_driv - elas_driv)
                dpdt[l,m,k][i] += dpi
                dpdt[l,m,k][j] += -dpi



@ti.kernel
def cut_off_phi_new(): # Keep φ in the range of 0 to 1
    for ID in ti.grouped(phi):
        phi_sum = 0.0

        for I in range(number_of_phases):
            new = phi[ID][I] + dpdt[ID][I]*dt[None]
            new = max(0.0, new)
            new = min(1.0, new)
            phi_new[ID][I] = new
            phi_sum += new

        for I in range(number_of_phases):
            phi_new[ID][I] /= phi_sum


@ti.kernel
def update_dc_atc_y(): # Antitrapping infused diffusion equation

    dudt.fill(0.0)
    for l,m,k in nfe:
        l_p, l_m, m_p, m_m, k_p, k_m = neighbor_index(l,m,k)
        C = l  , m  , k
        E = l_p, m  , k
        W = l_m, m  , k
        N = l  , m_p, k
        S = l  , m_m, k
        T = l  , m  , k_p
        B = l  , m  , k_m

        for i in range(nfe[C]):
            I = mfe[C][i]

            # 調和平均
            phi_e = 2 * phi[E][I]*phi[C][I] / (phi[E][I] + phi[C][I]) # scalar
            phi_w = 2 * phi[W][I]*phi[C][I] / (phi[W][I] + phi[C][I])
            phi_n = 2 * phi[N][I]*phi[C][I] / (phi[N][I] + phi[C][I])
            phi_s = 2 * phi[S][I]*phi[C][I] / (phi[S][I] + phi[C][I])
            phi_t = 2 * phi[T][I]*phi[C][I] / (phi[T][I] + phi[C][I])
            phi_b = 2 * phi[B][I]*phi[C][I] / (phi[B][I] + phi[C][I])

            for j in range(number_of_solutes):

                nab_e = ( ui[E][I,j]-ui[C][I,j])/dx # vector
                nab_w = (-ui[W][I,j]+ui[C][I,j])/dx
                nab_n = ( ui[N][I,j]-ui[C][I,j])/dy
                nab_s = (-ui[S][I,j]+ui[C][I,j])/dy
                nab_t = ( ui[T][I,j]-ui[C][I,j])/dz
                nab_b = (-ui[B][I,j]+ui[C][I,j])/dz
                
                JE = D[I,j]*phi_e*nab_e # vector
                JW = D[I,j]*phi_w*nab_w
                JN = D[I,j]*phi_n*nab_n
                JS = D[I,j]*phi_s*nab_s
                JT = D[I,j]*phi_t*nab_t
                JB = D[I,j]*phi_b*nab_b

                dudt[C][j] += ( (JE-JW)/dx + (JN-JS)/dy + (JT-JB)/dz )
        
        if antitraping: # Calculation of the antitrapping current

            if mfe[C][0] == 0 and nfe[C]>1: # Interface of the liquid

                for i in range(1,nfe[C]):
                    q = mfe[C][i]

                    for j in range(number_of_solutes):
                        
                        A = aij[0,q]/ti.sqrt(2*wij[0,q])
                        alpha_e = A * (2*(ui[E][0,j]-ui[E][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[E][0,j]-ui[E][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[E][0]*phi[C][0]/(phi[E][0]+phi[C][0]) * 2*phi[E][q]*phi[C][q]/(phi[E][q]+phi[C][q]) ) #[l+1/2,m,k]
                        alpha_w = A * (2*(ui[W][0,j]-ui[W][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[W][0,j]-ui[W][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[W][0]*phi[C][0]/(phi[W][0]+phi[C][0]) * 2*phi[W][q]*phi[C][q]/(phi[W][q]+phi[C][q]) ) #[l-1/2,m,k]
                        alpha_n = A * (2*(ui[N][0,j]-ui[N][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[N][0,j]-ui[N][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[N][0]*phi[C][0]/(phi[N][0]+phi[C][0]) * 2*phi[N][q]*phi[C][q]/(phi[N][q]+phi[C][q]) ) #[l,m+1/2,k]
                        alpha_s = A * (2*(ui[S][0,j]-ui[S][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[S][0,j]-ui[S][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[S][0]*phi[C][0]/(phi[S][0]+phi[C][0]) * 2*phi[S][q]*phi[C][q]/(phi[S][q]+phi[C][q]) ) #[l,m-1/2,k]
                        alpha_t = A * (2*(ui[T][0,j]-ui[T][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[T][0,j]-ui[T][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[T][0]*phi[C][0]/(phi[T][0]+phi[C][0]) * 2*phi[T][q]*phi[C][q]/(phi[T][q]+phi[C][q]) ) #[l,m,k+1/2]
                        alpha_b = A * (2*(ui[B][0,j]-ui[B][q,j])*(ui[C][0,j]-ui[C][q,j])/((ui[B][0,j]-ui[B][q,j])+((ui[C][0,j]-ui[C][q,j])))) * ti.sqrt( 2*phi[B][0]*phi[C][0]/(phi[B][0]+phi[C][0]) * 2*phi[B][q]*phi[C][q]/(phi[B][q]+phi[C][q]) ) #[l,m,k-1/2]

                        dpdt_e = ( (phi_new[E][0]+phi_new[C][0])/2 - (phi[E][0]+phi[C][0])/2 )/dt[None]
                        dpdt_w = ( (phi_new[W][0]+phi_new[C][0])/2 - (phi[W][0]+phi[C][0])/2 )/dt[None]
                        dpdt_n = ( (phi_new[N][0]+phi_new[C][0])/2 - (phi[N][0]+phi[C][0])/2 )/dt[None]
                        dpdt_s = ( (phi_new[S][0]+phi_new[C][0])/2 - (phi[S][0]+phi[C][0])/2 )/dt[None]
                        dpdt_t = ( (phi_new[T][0]+phi_new[C][0])/2 - (phi[T][0]+phi[C][0])/2 )/dt[None]
                        dpdt_b = ( (phi_new[B][0]+phi_new[C][0])/2 - (phi[B][0]+phi[C][0])/2 )/dt[None]
                        
                        p_c  = phi[C][0] - phi[C][q]
                        p_e  = phi[E][0] - phi[E][q]   
                        p_w  = phi[W][0] - phi[W][q]
                        p_n  = phi[N][0] - phi[N][q]
                        p_s  = phi[S][0] - phi[S][q]
                        p_t  = phi[T][0] - phi[T][q]
                        p_b  = phi[B][0] - phi[B][q]

                        p_en = phi[l_p,m_p,k  ][0] - phi[l_p,m_p,k  ][q]
                        p_es = phi[l_p,m_m,k  ][0] - phi[l_p,m_m,k  ][q]
                        p_et = phi[l_p,m  ,k_p][0] - phi[l_p,m  ,k_p][q]
                        p_eb = phi[l_p,m  ,k_m][0] - phi[l_p,m  ,k_m][q]

                        p_wn = phi[l_m,m_p,k  ][0] - phi[l_m,m_p,k  ][q]
                        p_ws = phi[l_m,m_m,k  ][0] - phi[l_m,m_m,k  ][q]
                        p_wt = phi[l_m,m  ,k_p][0] - phi[l_m,m  ,k_p][q]
                        p_wb = phi[l_m,m  ,k_m][0] - phi[l_m,m  ,k_m][q]

                        p_nt = phi[l  ,m_p,k_p][0] - phi[l  ,m_p,k_p][q]
                        p_nb = phi[l  ,m_p,k_m][0] - phi[l  ,m_p,k_m][q]
                        p_st = phi[l  ,m_m,k_p][0] - phi[l  ,m_m,k_p][q]
                        p_sb = phi[l  ,m_m,k_m][0] - phi[l  ,m_m,k_m][q]

                        phix_e = (p_e-p_c)/dx
                        phix_w = (p_c-p_w)/dx
                        phix_n = (p_en-p_wn+p_e-p_w) / (4*dx)
                        phix_s = (p_es-p_ws+p_e-p_w) / (4*dx)
                        phix_t = (p_et-p_wt+p_e-p_w) / (4*dx)
                        phix_b = (p_eb-p_wb+p_e-p_w) / (4*dx)

                        phiy_e = (p_en-p_es+p_n-p_s) / (4*dy)
                        phiy_w = (p_wn-p_ws+p_n-p_s) / (4*dy)
                        phiy_n = (p_n-p_c)/dy
                        phiy_s = (p_c-p_s)/dy
                        phiy_t = (p_nt-p_st+p_n-p_s) / (4*dy)
                        phiy_b = (p_nb-p_sb+p_n-p_s) / (4*dy)

                        phiz_e = (p_et-p_eb+p_t-p_b) / (4*dz)
                        phiz_w = (p_wt-p_wb+p_t-p_b) / (4*dz)
                        phiz_n = (p_nt-p_nb+p_t-p_b) / (4*dz)
                        phiz_s = (p_st-p_sb+p_t-p_b) / (4*dz)
                        phiz_t = (p_t-p_c)/dz
                        phiz_b = (p_c-p_b)/dz


                        nab_e = phix_e / ti.sqrt( phix_e**2 + phiy_e**2 + phiz_e**2 +1e-18 )
                        nab_w = phix_w / ti.sqrt( phix_w**2 + phiy_w**2 + phiz_w**2 +1e-18 )
                        nab_n = phiy_n / ti.sqrt( phix_n**2 + phiy_n**2 + phiz_n**2 +1e-18 )
                        nab_s = phiy_s / ti.sqrt( phix_s**2 + phiy_s**2 + phiz_s**2 +1e-18 )
                        nab_t = phiz_t / ti.sqrt( phix_t**2 + phiy_t**2 + phiz_t**2 +1e-18 )
                        nab_b = phiz_b / ti.sqrt( phix_b**2 + phiy_b**2 + phiz_b**2 +1e-18 )

                        JE = alpha_e*dpdt_e*nab_e
                        JW = alpha_w*dpdt_w*nab_w
                        JN = alpha_n*dpdt_n*nab_n
                        JS = alpha_s*dpdt_s*nab_s
                        JT = alpha_t*dpdt_t*nab_t
                        JB = alpha_b*dpdt_b*nab_b

                        dudt[C][j] += ((JE-JW)/dx + (JN-JS)/dy + (JT-JB)/dz)


@ti.kernel
def update_y(): # Evolution equation of the site fraction
    y_new.fill(0.0)
    for ID in ti.grouped(nf):
        for i in range(nf[ID]):
            I = mf[ID][i]
            f_grad = y.grad[ID] # first derivative of the chemical free energy
            for a in range(y_phase_ID[I], y_phase_ID[I+1]): 

                for j in range(number_of_solutes):
                    if nf[ID] == 1 and y_num[I] == 1: # If neither the interface nor the sublattice phase
                        y_new[ID][a,j] = y[ID][a,j] + dudt[ID][j]*dt[None]
                    else:
                        Sa     = y_S[a]
                        fyya   = fyy[ID][a,j]
                        fya    = f_grad[a,j]
                        top    = dudt[ID][j]*dt[None]
                        bottom = 0.0
                    
                        for ii in range(nf[ID]):
                            II = mf[ID][ii]

                            for b in range(y_phase_ID[II], y_phase_ID[II+1]):
                                Sb   = y_S[b]
                                yb   = y[ID][b,j]
                                top += -(phi_new[ID][II]-phi[ID][II])*Sb*yb

                                fyyb =  fyy[ID][b,j]
                                fyb  = f_grad[b,j]
                                if phi[ID][I] > 0 and phi[ID][II] > 0:
                                    top    += -phi_new[ID][II]*Sb*(fya*Sb-fyb*Sa)/(number_of_solutes*fyyb*Sa) 
                                bottom +=  phi_new[ID][II]*Sb*Sb/Sa*fyya/fyyb
                        
                        del_y = top/bottom
                        y_new[ID][a,j] = y[ID][a,j] + del_y


@ti.kernel
def update_field(cooling_rate:float): # Update T,phi and y
    nstep[None] += 1
    solidification_time[None] += dt[None]

    dT = dt[None]*cooling_rate
    for ID in ti.grouped(phi):
        T[ID] += dT

    for ID in ti.grouped(phi):
        phi[ID] = phi_new[ID]        
        y[ID] = y_new[ID]

    


@ti.kernel
def regulate_dt(max_dp:float): # After initial placement, the local minimisation condition is reached by computing with smaller time steps.

    dt_regulated = dt_max
    for ID in ti.grouped(dpdt):
        for i in range(number_of_phases):

            ti.atomic_min(dt_regulated, max_dp/(abs(dpdt[ID][i])+1e-20))
        
    dt[None] = dt_regulated




def update_eij_elast(): # cubic
  # 1. calculation of the eigne strain
  eigen_strain = eigen_coeff*(1.0 - phi.to_numpy()[:,:,0,0]) # xx

  # 2. calculation of the homogeneous strain
  homo_strain = np.sum(eigen_strain) / (nx*ny*nz)

  # 3. Fourier transform of the eigen strain
  eigen_strain_inv = np.fft.fftn(eigen_strain) # xx=yy, xy=0

  # 4. calculation of the inhomogeneous strain
  zai = (c11-c12-2*c44) / c44
  A = c44 * (c11 + zai*(c11+c12)*kxx*kyy)
  inhomo_strain_inv[0] = (c11+2*c12)/A * (c44*kxx + (c11-c12-2*c44)*kxx*kyy) * eigen_strain_inv # εxx
  inhomo_strain_inv[1] = (c11+2*c12)/A * (c44*kyy + (c11-c12-2*c44)*kxx*kyy) * eigen_strain_inv # εyy
  inhomo_strain_inv[2] = (c11+2*c12)/2 * kx*ky/A * (c11-c12) * eigen_strain_inv # εxy

  # 5. Inverse fourier transform of the inhomogeneous strain
  inhomo_strain[0] = np.real(np.fft.ifftn(inhomo_strain_inv[0]))
  inhomo_strain[1] = np.real(np.fft.ifftn(inhomo_strain_inv[1]))
  inhomo_strain[2] = np.real(np.fft.ifftn(inhomo_strain_inv[2]))

  # 6. calculation of the elastic strain and stress
  elastic_strain[0] = homo_strain + inhomo_strain[0] - eigen_strain
  elastic_strain[1] = homo_strain + inhomo_strain[1] - eigen_strain
  elastic_strain[2] = 0.0         + inhomo_strain[2] - 0.0
  stress[0] = c11*elastic_strain[0] + c12*elastic_strain[1]
  stress[1] = c12*elastic_strain[0] + c11*elastic_strain[1]
  stress[2] = c44*elastic_strain[2] * 2.0

  # 7. calculation of the driving force for the interface movement due to the elastic strain
  eij_elast[:,:,0] = - eigen_coeff * (c11*elastic_strain[0] + c12*elastic_strain[1]) \
                     - eigen_coeff * (c12*elastic_strain[0] + c11*elastic_strain[1])
  eij_elast_taichi.from_numpy(eij_elast)



def step(cooling_rate, regulate=False, max_dp=0.01):
    update_nfmf()
    interp_y()
    update_f()
    y.grad.fill(0)
    f.grad.fill(1)
    update_f.grad()
    update_fyy()
    update_ui()
    update_eij()
    if elastic: update_eij_elast()
    if antitraping: update_mij()
    update_dpdt()
    if regulate: regulate_dt(max_dp)
    cut_off_phi_new()
    update_dc_atc_y()
    update_y()
    update_field(cooling_rate)
# ----------------------------↑Definition of functions↑---------------------------------------------



# ----------------------------↓Functions for initialize↓---------------------------------------------
@ti.kernel
def set_round(i:int, r_nuclei:float, x_nuclei:float, y_nuclei:float, z_nuclei:float): #　Place phase i of radius r at coordinate [x_nuclei, y_nuclei, z_nuclei]
    for l in range(nx):
        for m in range(ny):
            for k in range(nz):
                r = ti.sqrt( (l *dx-x_nuclei*dx)**2 + (m*dy-y_nuclei*dy)**2 + (k*dz-z_nuclei*dz)**2 ) - r_nuclei # Distance between [l,m,k] and [x_nuclei, y_nuclei, z_nuclei]
                tmp = ti.sqrt(2.*wij[0,i])/aij[0,i]*r
                phi_tmp = 0.5*(1.-ti.sin(tmp))
                if tmp >= pi/2.:
                    phi_tmp=0.
                if tmp <= -pi/2.:
                    phi_tmp=1.
                if 0. < phi_tmp < 1.: 
                    nf_tmp = nf[l,m,k]+1
                    nf[l,m,k] = nf_tmp
                    mf[l,m,k][nf_tmp-1] = i
                    for ii in range(number_of_phases):
                        if phi[l,m,k][ii] > 0:
                            phi[l,m,k][ii] -= phi_tmp
                    phi[l,m,k][i] += phi_tmp 
                    if phi[l,m,k][0] < 0.:
                        phi[l,m,k][0] = 0.
                if phi_tmp >= 1.: 
                    nf_tmp = 1
                    nf[l,m,k] = nf_tmp
                    mf[l,m,k][0] = i
                    for ii in range(number_of_phases):
                        if phi[l,m,k][ii] > 0:
                            phi[l,m,k][ii] = 0.0
                    phi[l,m,k][i] = phi_tmp            



def check_overlap(circles, new_circle, min_distance): # Check if the new_circle overlaps the other circles
    for (pos_x, pos_y) in circles:
        dx = min(abs(pos_x - new_circle[0]), nx - abs(pos_x - new_circle[0]))
        dy = min(abs(pos_y - new_circle[1]), ny - abs(pos_y - new_circle[1]))
        distance = np.sqrt(dx**2 + dy**2)
        if distance < min_distance:
            return False
    return True
def generate_non_overlapping_positions(num_positions, min_distance): # Generate coordinates where the minimum distance between each other is min_distance
    positions = []
    while len(positions) < num_positions:
        pos_x = np.random.uniform(0, nx)
        pos_y = np.random.uniform(0, ny)
        new_position = (pos_x, pos_y)
        if check_overlap(positions, new_position, min_distance):
            positions.append(new_position)
    return positions


fluctuation_field = ti.field(dtype=float, shape=(nx,ny,nz))
l12_fraction = 0.4 # initial fraction of l12 phases
@ti.kernel
def fluctuate_phi(): # Flucutuate phi to generate l12 phases
    for ID in ti.grouped(phi):
        random_value = ti.random()
        fluctuation_field[ID] = random_value
    sum = 0.0
    for ID in ti.grouped(phi):
        sum += fluctuation_field[ID]
    average = sum/(nx*ny*nz)
    for ID in ti.grouped(phi):
        phi[ID][0] -= fluctuation_field[ID]/average*l12_fraction
        phi[ID][1] += fluctuation_field[ID]/average*l12_fraction


@ti.kernel
def initialize_taichi_field(): # Initialize taichi field
    nstep[None] = 0
    solidification_time[None] = 0.0
    T.fill(Tini)
    dt[None] = dt_max
    phi.fill(0.0)
    phi_new.fill(0.0)
    y.fill(0.0)
    y_new.fill(0.0)
    eij.fill(0.0)
    dudt.fill(0.0)
    dpdt.fill(0.0)
    eij_elast_taichi.fill(0.0)

    for ID in ti.grouped(nf):
        phi[ID][0] = 1.0
        
        for a in range(number_of_sublattices):
            for j in range(number_of_solutes):
                y[ID][a,j] = yini[a,j]
    
    for ID in ti.grouped(phi):
        for i in range(number_of_phases):
            phi_new[ID][i] = phi[ID][i]
        y_new[ID] = y[ID]



def check_compile_time(): # Check compile time
    start = time.time()
    update_fyy()
    print(f'compilation time of update_fyy() is {(time.time() - start):.2f}')
    start = time.time()
    step(cooling_rate=0.0, regulate=True, max_dp=0.0001)
    print(f'compilation time of step()       is {(time.time() - start):.2f}')

def init(): # Initialize
    check_compile_time()
    np.random.seed(0)
    initialize_taichi_field()
    exec(initialize_text) # Set nucleis
    start_dp, end_dp, division_steps = 0.00001, 0.01, 3000
    for i in range(division_steps):
        relaxation_dp = start_dp + (end_dp-start_dp)/division_steps*i
        step(cooling_rate=0.0, regulate=True, max_dp=relaxation_dp) # Initially, the time increment is reduced to stabilize the calculation
    nstep.fill(0) # Reset nstep
    solidification_time.fill(0.0) # Reset solidification_time
# ----------------------------↑Functions for initialize↑---------------------------------------------





# ----------------------------↓Functions for visualization↓---------------------------------------------
@ti.kernel
def update_u(): # Calculate composition
    u.fill(0.0)
    for ID in ti.grouped(nfe):
        for i in range(nfe[ID]):
            I = mfe[ID][i]
            for a in range(y_phase_ID[I], y_phase_ID[I+1]):
                for j in range(number_of_solutes):
                    uiaj = y[ID][a,j]*y_S[a]
                    u[ID][j] += phi[ID][I]*uiaj

def prepare_canvas(figsize): # prepare plt.figre
    fig = plt.figure(figsize=figsize)
    fig.set_dpi(100)
    plt.subplots_adjust(wspace=0.3)   

def adjust_subplot(): # adjust plt.imshow
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(aspect=10, pad=0.1,  location='bottom')  

def visualize(z=0): # Visualize phase distribution, number of phases and composition (cross-section at z)
    update_nfmf()
    update_u()
    phi_numpy = np.transpose(phi.to_numpy(), (3,0,1,2))
    ci_numpy  = np.transpose(ui.to_numpy(), (0,4,1,2,3))
    gi        = np.argmax(phi_numpy, axis=0)

    figsize = (12,4)
    prepare_canvas(figsize)

    # visualize phase distribution
    plt.subplot(1,3+number_of_solutes, 1)
    plt.imshow(gi[:,:,z], cmap='jet', vmin=0, vmax=number_of_phases-1)
    plt.title('phase ID')
    adjust_subplot()
    # visualize number of phases
    plt.subplot(1,3+number_of_solutes, 2)
    plt.imshow(nfe.to_numpy()[:,:,z], cmap='bwr', vmin=1, vmax=4)
    plt.title('num phases')
    adjust_subplot()
    # visualize composition of solvent
    plt.subplot(1,3+number_of_solutes, 3)
    plt.imshow(1-np.sum(u.to_numpy()[:,:,z,:], axis=-1))
    plt.title(f'{solvent}')
    adjust_subplot()
    # visualize solute
    for i in range(number_of_solutes): 
        plt.subplot(1, 3+number_of_solutes, i+1+3)
        plt.imshow(u.to_numpy()[:,:,z,i])
        plt.title(f'{solutes[i]}')
        adjust_subplot()
    plt.show()
 
# ----------------------------↑Functions for visualization↑---------------------------------------------






# ----------------------------↓Function for error measurement↓---------------------------------------------
counts         = ti.field(int  , shape=(nx,ny,nz)) # number of error measurement
sum_error      = ti.field(float, shape=(nx,ny,nz)) # sum of the relative error of the diffusion potential
sum_error_Jmol = ti.field(float, shape=(nx,ny,nz)) # sum of the error of the diffusion potential

@ti.kernel
def evaluate_error(): # evaluate error
    for ID in ti.grouped(nfe):
        for i in range(nfe[ID]):
            I = mfe[ID][i]
            for a in range(y_phase_ID[I], y_phase_ID[I+1]):
                fy = y.grad[ID]

                # Measure the error with the sublattice of another phase
                if nfe[ID] > 1: 
                    for ii in range(i+1, nfe[ID]):
                        II = mfe[ID][ii] 
                        for b in range(y_phase_ID[II], y_phase_ID[II+1]):
                            for j in range(number_of_solutes):
                                counts[ID]         += 1
                                sum_error[ID]      += abs((fy[a,j]/y_S[a]-fy[b,j]/y_S[b])/(fy[a,j]/y_S[a]))
                                sum_error_Jmol[ID] += abs(fy[a,j]/y_S[a]-fy[b,j]/y_S[b])

                # Measure the error with the sublattice of the same phase
                if y_num[I] > 1:
                    for b in range(a+1, y_phase_ID[I+1]):
                        for j in range(number_of_solutes):
                            counts[ID]         += 1
                            sum_error[ID]      += abs((fy[a,j]/y_S[a]-fy[b,j]/y_S[b])/(fy[a,j]/y_S[a]))
                            sum_error_Jmol[ID] += abs(fy[a,j]/y_S[a]-fy[b,j]/y_S[b])

# ----------------------------↑Function for error measurement↑---------------------------------------------
