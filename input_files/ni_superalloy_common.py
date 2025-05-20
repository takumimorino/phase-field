import numpy as np
import taichi as ti
import os
exec(f'from input_files.{os.getenv("import_file")} import *')
if 1 == 2: from input_files.ni_superalloy_3 import * # dummy

# -----------------------------------------------Definition of the system-------------------------------------------------------
solvent         = 'NI' # name of solvent
phases          = ['FCC_A1', 'L12', 'L12', 'L12', 'L12']                  # name of each phase
set_y_num       = np.array([1,2,2,2,2])                                   # number of sublattices for each phase
set_y_S         = np.array([1.0, 1/4, 3/4, 1/4, 3/4, 1/4, 3/4, 1/4, 3/4]) # Stoichiometory number of each sublattice
set_y_phase_ID  = np.array([0,1,3,5,7,9])                                 # Relationship between phase ID and sublattice ID (FCC_A1→0, L12→1to2, L12→3to4, L12→5to6, L12→7to8）

# -----------------------------------------------Calculation condition-------------------------------------------------------
Tini                 = 1400.0                 # initial temperature [K]
cooling_rate         = [0.0]                  # Cooling rate [K/s]
nx, ny, nz           = 128, 128, 1            # number of computational grids
dx, dy, dz           = 5.0e-9, 5.0e-9, 5.0e-9 # spacing of computational grids [m] 
delta                = 4.0 * dx               # thickness of diffuse interface
steps                = [30000]                # Total time step
boundary_condition_x = 'periodic'             # boundary condition (periodic or zero_Neumann)
boundary_condition_y = 'periodic'             # boundary condition (periodic or zero_Neumann)
boundary_condition_z = 'periodic'             # boundary condition (periodic or zero_Neumann)
anisotoropy          = False                  # Incorpolate anisotoropy?
nu                   = 0.03                   # strength of the anisotoropy
antitraping          = False                  # Incorpolate anti-trapping current?
elastic              = True                   # Incorpolate elastic energy?
initialize_text = """
positions = generate_non_overlapping_positions(num_positions=100, min_distance=8)
for (pos_x, pos_y) in positions:
    radius = 0.1 + np.random.rand()*1.0 # 1~3の核サイズ
    nuc_id = np.random.randint(1,5)
    set_round(nuc_id, dx*radius, pos_x, pos_y, 0)
"""                                           # A code to set nuclei

# -----------------------------------------------Material parameters-------------------------------------------------------
set_Vm                    = np.full(len(phases), 1e-5)                 # molar volume of each phase [m3/mol]
set_sigma                 = np.full((len(phases), len(phases)), 0.05)  # interfacial energy[J/m2] between γ/γ and γ'/γ'
set_sigma[0,:]            = 0.02                                       # interfacial energy[J/m2] between γ/γ'
set_sigma[:,0]            = 0.02                                       # interfacial energy[J/m2] betewen γ/γ'
np.fill_diagonal(set_sigma, 0)
set_D                     = np.full((len(phases),len(solutes)), 1e-16) # diffusion coefficient [m2/s]
set_yini                  = np.array([uini_ave, yini_L12[0], yini_L12[1], yini_L12[0], yini_L12[1], yini_L12[0], yini_L12[1], yini_L12[0], yini_L12[1]]) # initial site fraction
set_interface_mobility    = np.zeros((len(phases), len(phases)))       # interface mobility
PF_mobility               = 5e-9                                       # value of the phase-field mobility
set_interface_mobility[:] = PF_mobility / (np.pi**2/(4*delta))         # This is interface mobility. The value here is 4.0528e-17.
c11 = 250.8e9                                                          # elastic constant
c12 = 150.0e9                                                          # elastic constant
c44 = 123.5e9                                                          # elastic constant
eigen_coeff = -0.006                                                   # Eigen strain between γ and γ'