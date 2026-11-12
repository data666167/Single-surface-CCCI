import sys
import os
from os.path import abspath, join, dirname, basename
import numpy as np
import pandas as pd

# ev to cm conversion factor 
conv = 8065.54

dataset = 'CH2NH'
if dataset == 'CH3CN':
    csv_paths = ['../Results/CC_results/CH3CN_trial_1000/tabulations/ACF_ABS_CC_ch3cn_ev_h4_Z4_T1000.0_normalized_t200.csv',
                 '../Results/CC_results/CH3CN_trial_1000/tabulations/ACF_ABS_CC_ch3cn_ev_h4_Z4_T1000.0_normalized_t500.csv',
                 '../Results/CC_results/CH3CN_trial_1000/tabulations/ACF_ABS_CC_ch3cn_ev_h4_Z4_T1000.0_normalized_t1000.csv',]
    output_path = '../Results/CC_results/ch3cn_tabulated_results/test_comp_d5_T1000.csv'
if dataset == 'CH2NH':
    csv_paths = ['../Results/CC_results/CH2NH_trial_1000/tabulations/ACF_ABS_CC_ch2nh_evp_h4_Z4_T1000.0_normalized_t200.csv',
                 '../Results/CC_results/CH2NH_trial_1000/tabulations/ACF_ABS_CC_ch2nh_evp_h4_Z4_T1000.0_normalized_t500.csv',
                 '../Results/CC_results/CH2NH_trial_1000/tabulations/ACF_ABS_CC_ch2nh_evp_h4_Z4_T1000.0_normalized_t1000.csv',]
    output_path = '../Results/CC_results/ch2nh_tabulated_results/test_comp_d5_T1000.csv'

 

df_ref = pd.read_csv(csv_paths[0])
ex_ref = df_ref.loc[0,'reference eigenvalues'] 

# excitations relative reference groundstate
df_ref.loc[1:,'reference eigenvalues'] += -ex_ref 
# ev to wavenumber
df_ref['reference eigenvalues'] *= conv 

comp_dict = {'reference eigenvalues':df_ref['reference eigenvalues']}
for csv_path in csv_paths:
    # excitations relative reference groundstate
    df = pd.read_csv(csv_path)
    df.loc[1:, 'nearest extracted eigenvalues'] += -ex_ref 
    
    # ev to wavenumber
    df['nearest extracted eigenvalues'] *= conv
    comp_dict[os.path.splitext(basename(csv_path))[0]] = df['nearest extracted eigenvalues']
    df

comp_df = pd.DataFrame(comp_dict)
comp_df.to_csv(output_path, index=False)