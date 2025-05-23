# Program and data for "An explicit integration approach for predicting the microstructures of multicomponent alloys"

This repository contains the program code used to perform the calculations described in the paper:

> Takumi Morino, Machiko Ode and Shoich Hirosawa, "An explicit integration approach for predicting the microstructures of multicomponent alloys", Nature Communications

## Files in the Repository

- 'main.py': The main script to execute the calculations.
  You can select calculation system by changing `os.environ['import_file']`.
  The calculation system includes
  - al-dendrite in 3 components system
  - al-dendrite in 6 components system
  - al-dendrite in 9 components system
  - al-dendrite in 12 components system

  - ni-dendrite in 3 components system
  - ni-dendrite in 6 components system
  - ni-dendrite in 9 components system
  - ni-dendrite in 12 components system
  - ni-dendrite in 15 components system

  - fe-dendrite in 3 components system
  - fe-dendrite in 6 components system
  - fe-dendrite in 9 components system
  - fe-dendrite in 12 components system
  - fe-dendrite in 15 components system
  - fe-dendrite in 18 components system
  - fe-dendrite in 20 components system

  - ni-superalloy in 3 components system
  - ni-superalloy in 6 components system
  - ni-superalloy in 9 components system
  - ni-superalloy in 12 components system

- 'functions.py': Functions.
- 'input_files': Calculation and numerical paramters.
- 'data_for_figures': Data for figures in the paper.
- 'TDBConvTakumi.ipynb': A system that convert TDB files into python style free energy functions.
- 'README.md': Documentation for the program.
