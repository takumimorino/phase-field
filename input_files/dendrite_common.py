import numpy as np
import taichi as ti
import os
exec(f'from input_files.{os.getenv("import_file")} import *')
if 1 == 2: from input_files.ni_dendrite_3 import * # dummy

if 'ni_dendrite' in os.getenv("import_file"):
    solvent = 'NI'
    Tini    = 1700.0
if 'al_dendrite' in os.getenv("import_file"):
    solvent = 'AL'
    Tini    = 900.0
if 'fe_dendrite' in os.getenv("import_file"):
    solvent = 'FE'
    Tini    = 1700.0

# -----------------------------------------------Definition of the system-------------------------------------------------------
phases          = ['LIQUID', 'FCC_A1'] # name of each phase           
set_y_num       = np.array([1,1])      # number of sublattices for each phase
set_y_S         = np.array([1.0, 1.0]) # Stoichiometory number of each sublattice
set_y_phase_ID  = np.array([0,1,2])    # Relationship between phase ID and sublattice ID (FCC_A1→0, L12→1to2, L12→3to4, L12→5to6, L12→7to8）

# -----------------------------------------------Calculation condition-------------------------------------------------------
cooling_rate         = [-50.0]                # Cooling rate [K/s]
nx, ny, nz           = 128, 128, 1            # number of computational grids
dx, dy, dz           = 0.1e-6, 0.1e-6, 0.1e-6 # spacing of computational grids [m] 
delta                = 4.0 * dx               # thickness of diffuse interface
steps                = [30000]                # Total time step
boundary_condition_x = 'zero_Neumann'         # boundary condition (periodic or zero_Neumann)
boundary_condition_y = 'zero_Neumann'         # boundary condition (periodic or zero_Neumann)
boundary_condition_z = 'zero_Neumann'         # boundary condition (periodic or zero_Neumann)
anisotoropy          = True                   # Incorpolate anisotoropy?
nu                   = 0.03                   # strength of the anisotoropy
antitraping          = True                   # Incorpolate anti-trapping current?
elastic              = False                  # Incorpolate elastic energy?
initialize_text = """
set_round(1, dx*5, 0, 0, 0)
"""                                           # A code to set nuclei

# -----------------------------------------------Material parameters-------------------------------------------------------
set_Vm                    = np.full(len(phases), 1e-5)                 # molar volume of each phase [m3/mol]
set_sigma                 = np.full((len(phases), len(phases)), 0.2)   # interfacial energy[J/m2]
np.fill_diagonal(set_sigma, 0)
set_D                     = np.full((len(phases),len(solutes)), 1e-13) # diffusion coefficient [m2/s] in solid
set_D[0] = 1e-9 # diffusion coefficient [m2/s] in liquid
set_yini                  = np.array([uini_ave, yini_FCC_A1[0]])       # initial site fraction
set_interface_mobility    = np.zeros((len(phases), len(phases)))       # interface mobility
c11 = 250.8e9                                                          # elastic constant
c12 = 150.0e9                                                          # elastic constant
c44 = 123.5e9                                                          # elastic constant
eigen_coeff = -0.006                                                   # Eigen strain between γ and γ'

