import numpy as np
import time
import os
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)
os.environ['import_file'] = 'al_dendrite_3'  # 3~12
# os.environ['import_file'] = 'ni_dendrite_3'  # 3~15
# os.environ['import_file'] = 'fe_dendrite_20'  # 3~20
# os.environ['import_file'] = 'ni_superalloy_9'  # 3~12
from functions_cupsul import *

init()

ti.sync()
start = time.perf_counter()
for i in range(steps[0]+1):
    step(cooling_rate=cooling_rate[0])
    if i % 100 == 0:
        evaluate_error()

ti.sync()
end = time.perf_counter()
average_absolute_error = np.sum(sum_error.to_numpy())/np.sum(counts.to_numpy())
average_error = np.sum(sum_error_Jmol.to_numpy())/np.sum(counts.to_numpy())
print(f"calculation time       : {end-start}")
print(f"average absolute error : {average_absolute_error}")
print(f"average error          : {average_error}")
visualize()


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