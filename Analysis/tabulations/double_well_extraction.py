import numpy as np 
import matplotlib.pyplot as plt
from signal_processing import signal_process as sp
import pandas as pd
# extract spectra


cc_acf_paths = [
'../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-7/ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt',
'../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_7/ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt'
]
ED_acf_paths =[
'../Results/ED_results/double_well_tests/results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_H47d.txt',    
'../Results/ED_results/double_well_tests/results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_H47e.txt'
]

output_directory = '../Results/ED_results/double_well_tests/tabulations/extracted_energies.csv'

extracted_values = {}
idx = 1 
for idx in range(2):
    cc_data  = np.loadtxt(cc_acf_paths[idx],skiprows=1)
    ED_data  = np.loadtxt(ED_acf_paths[idx],skiprows=1)

    cc_time = cc_data[:,0]
    ED_time = ED_data[:,0]

    cc_acf =  cc_data[:,1] + 1j*cc_data[:,2]
    ED_acf =  ED_data[:,1] + 1j*ED_data[:,2]

    prop_cc_time =  cc_time
    prop_ED_time =  ED_time

    prop_cc_acf =  cc_acf
    prop_ED_acf =  ED_acf


    opt_cc, raw_cc = sp.peak_detector(prop_cc_time/.658211957,prop_cc_acf,peak_height_minimum=0.0005,energy_threshold_ratio=0.05)
    opt_ED, raw_ED = sp.peak_detector(prop_ED_time/.658211957,prop_ED_acf,peak_height_minimum=0.0005,energy_threshold_ratio=0.01)
    if idx == 0: 
        disp = -7
    else:
        disp = 7 
    extracted_values[f'extracted CC values for disp {disp}'] = pd.Series(np.sort(opt_cc).real)
    extracted_values[f'extracted ED values for disp {disp}'] = pd.Series(np.sort(opt_ED).real)
    
    
df = pd.DataFrame(extracted_values)
df.to_csv(output_directory,index=False)

