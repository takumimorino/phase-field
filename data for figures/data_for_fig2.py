# data for Figure 2

import numpy as np

number_of_component    = np.array([3,6,9,12,15,18,20])
data_for_fig2 = {
    'al_dendrite': {
        'error'     : np.array([5.32e-7, 4.01e-6, 7.45e-6, 7.89e-6, np.nan,np.nan,np.nan]),
        'errbar'    : np.array([[4.00e-8, 1.04e-7, 1.36e-7, 1.25e-7,np.nan,np.nan,np.nan],
                                [1.73e-7, 7.79e-6, 2.70e-5, 2.71e-5,np.nan,np.nan,np.nan]]),
        'error_Jmol': np.array([0.0241, 0.294, 0.397, 0.498, np.nan,np.nan,np.nan]),
        'errbar_Jmol':np.array([[0.00274, 0.00506, 0.00701, 0.0126,np.nan,np.nan,np.nan],
                                [0.00483, 0.592  , 0.929  , 1.41  ,np.nan,np.nan,np.nan]]),
        'time'      : np.array([25,30,35,42,np.nan,np.nan,np.nan])
    },
    'ni_dendrite': {
        'error'     : np.array([5.53e-7, 1.70e-6, 2.96e-6, 2.80e-6, 5.16e-6,np.nan,np.nan]),
        'errbar'    : np.array([[7.53e-9, 1.28e-8, 8.14e-8, 2.91e-8, 1.62e-7, np.nan,np.nan],
                                [4.89e-7, 2.40e-6, 3.71e-6, 4.41e-6, 1.05e-5, np.nan,np.nan]]),
        'error_Jmol': np.array([0.0476, 0.107, 0.233, 0.269, 0.484,np.nan,np.nan]),
        'errbar_Jmol':np.array([[0.000854, 0.00104, 0.00643, 0.00451, 0.0164,np.nan,np.nan],
                                [0.0297  , 0.143  , 0.470  , 0.392  , 1.01  ,np.nan,np.nan]]),
        'time'      : np.array([25,29,33,41,48,np.nan,np.nan])
    },
    'fe_dendrite': {
        'error'     : np.array([6.12e-6, 3.74e-6, 6.54e-6, 8.20e-6, 2.83e-5, 1.89e-5, 1.16e-5]),
        'errbar'    : np.array([[2.58e-7, 6.22e-8, 5.29e-8, 8.39e-8, 1.85e-7, 1.74e-7, 1.44e-7],
                                [9.81e-6, 8.20e-6, 1.67e-5, 2.65e-5, 9.65e-5, 6.57e-5, 3.52e-5]]),
        'error_Jmol': np.array([0.0477, 0.131, 0.282, 0.300, 0.504, 0.535, 0.604]),
        'errbar_Jmol':np.array([[0.00297, 0.00178, 0.00427, 0.00613, 0.0160, 0.0175, 0.0141],
                                [0.0950 , 0.306  , 0.692  , 0.737  , 1.223 , 1.27  , 1.44  ]]),
        'time'      : np.array([25, 29, 34, 41, 51, 56, 63])
    },
    'ni_superalloy': {
        'error'     : np.array([1.11e-6, 9.93e-6, 8.90e-6, 1.07e-5, np.nan,np.nan,np.nan]),
        'errbar'    : np.array([[4.24e-9, 8.51e-9, 2.26e-8, 2.33e-8,np.nan,np.nan,np.nan],
                                [3.45e-6, 3.38e-5, 2.89e-5, 3.34e-5,np.nan,np.nan,np.nan]]),
        'error_Jmol': np.array([0.0369, 0.0903, 0.132, 0.184, np.nan,np.nan,np.nan]),
        'errbar_Jmol':np.array([[0.000246, 0.000422, 0.000988, 0.00128,np.nan,np.nan,np.nan],
                                [0.110   , 0.281   , 0.433   , 0.578  ,np.nan,np.nan,np.nan]]),
        'time'      : np.array([81, 147, 207, 308, np.nan,np.nan,np.nan])
    }
}