from signal_processing import excite_extraction as extr
import sys
import os
from os.path import abspath, join, dirname, basename
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# extract eigvals
evals_ch2nh = [
1.077622,  1.206751,  1.210890,  1.218982,  1.242811,  1.256295,  1.279382,  1.335022,
1.340094,  1.341986,  1.347735,  1.351282,  1.356904,  1.368162,  1.376041,  1.384091,
1.384351,  1.389494,  1.396718,  1.405429,  1.407302,  1.412129,  1.419802,  1.420890,
1.435755]

evals_ch3cn =[
1.21968, 1.26444, 1.30934, 1.30942, 1.33135, 1.34790, 1.35440, 1.35455, 
1.37588, 1.39189, 1.39260, 1.39263, 1.39297, 1.40358, 1.42056, 1.43659, 
1.43745, 1.43754, 1.43786, 1.44101, 1.44521, 1.44834, 1.44835, 1.44839, 
1.45916, 1.46540, 1.47473, 1.47574, 1.48146,  
]

# ev to cm conversion factor 
conv = 8065.54

dataset = 'CH2NH'
if dataset == 'CH3CN':
    base_dir = Path('../Results/CC_results/CH3CN_single_mode/results')
    evals  = evals_ch3cn
    z_2_results = '*model_ch3cn_ev_rank_4_Z_2*'
    z_4_results = '*model_ch3cn_ev_rank_4_Z_4*'
    output_path_Z2 = '../Results/CC_results/CH3CN_single_mode/tabulations/Z2'
    output_path_Z4 = '../Results/CC_results/CH3CN_single_mode/tabulations/Z4'
    
if dataset == 'CH2NH':
    base_dir = Path('../Results/CC_results/CH2NH_single_mode/results')
    evals  = evals_ch2nh
    z_2_results = '*model_ch2nh_evp_rank_4_Z_2*'
    z_4_results = '*model_ch2nh_evp_rank_4_Z_4*'
    output_path_Z2 = '../Results/CC_results/CH2NH_single_mode/tabulations/Z2'
    output_path_Z4 = '../Results/CC_results/CH2NH_single_mode/tabulations/Z4'


    
result_dirs_Z2 = [path for path in base_dir.glob(z_2_results) if path.is_dir()] 
acf_paths_Z2 = []
for result_dir in result_dirs_Z2:
  for path in result_dir.glob('*norm*'):
    if path.is_file():
      acf_paths_Z2.append(path)

result_dirs_Z4 = [path for path in base_dir.glob(z_4_results) if path.is_dir()] 
acf_paths_Z4 = [] 
for result_dir in result_dirs_Z4:
  for path in result_dir.glob('*norm*'):
    if path.is_file():
      acf_paths_Z4.append(path)
      

for path in acf_paths_Z2:
  df ,_ ,_ = extr.extract_freqeuncy(path,final_time=500,ref_eig_vals=evals,prominence=(0.005))
  file_name = os.path.splitext(basename(path))[0]+'.csv'
  output_file_path = join(output_path_Z2,file_name) 
  df.to_csv(output_file_path, index=False)
  # breakpoint()


for path in acf_paths_Z4:
  df ,_ ,_ = extr.extract_freqeuncy(path,final_time=500,ref_eig_vals=evals,prominence=(0.005))
  file_name = os.path.splitext(basename(path))[0]+'.csv'
  output_file_path = join(output_path_Z4,file_name) 
  df.to_csv(output_file_path, index=False)

