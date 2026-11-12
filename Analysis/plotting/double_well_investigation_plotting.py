import sys
import os
from os.path import abspath, join, dirname, basename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# CC paths
CC_dirs = [
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-7',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-14',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_0',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_7',
    ]

CC_filenames =[
    'exp_H_H47b_h4_Z4_T200.0_normalized.txt',
    'norm_H47b_h4_Z4_T200.0_normalized.txt',
    'exp_q_H47b_h4_Z4_T200.0_normalized.txt',
    'exp_q_2_H47b_h4_Z4_T200.0_normalized.txt',
    ]

CC_norm_dirs =[
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-7_ext_norm',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-14_ext_norm',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_0_ext_norm',
    '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_7_ext_norm',
    ]


# ED paths
ED_energy = [
    '../Results/ED_results/double_well_tests/results/model_H47b_rank_4_basis_400_time_200/ACF_ABS_CC_exp_H_H47b.txt',
    '../Results/ED_results/double_well_tests/results/model_H47c_rank_4_basis_400_time_200/ACF_ABS_CC_exp_H_H47c.txt',
    '../Results/ED_results/double_well_tests/results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_exp_H_H47d.txt',
    '../Results/ED_results/double_well_tests/results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_exp_H_H47e.txt',
    '../Results/ED_results/double_well_tests/results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_exp_H_H47f.txt',
    ]
ED_norm = [
    '../Results/ED_results/double_well_tests/results/model_H47b_rank_4_basis_400_time_200/ACF_ABS_CC_norm_H47b.txt',
    '../Results/ED_results/double_well_tests/results/model_H47c_rank_4_basis_400_time_200/ACF_ABS_CC_norm_H47c.txt',
    '../Results/ED_results/double_well_tests/results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_norm_H47d.txt',
    '../Results/ED_results/double_well_tests/results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_norm_H47e.txt',
    '../Results/ED_results/double_well_tests/results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_norm_H47f.txt',
    ]

ED_q = [
   '../Results/ED_results/double_well_tests/results/model_H47b_rank_4_basis_400_time_200/ACF_ABS_CC_q_H47b.txt',
   '../Results/ED_results/double_well_tests/results/model_H47c_rank_4_basis_400_time_200/ACF_ABS_CC_q_H47c.txt',
   '../Results/ED_results/double_well_tests/results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_q_H47d.txt',
   '../Results/ED_results/double_well_tests/results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_q_H47e.txt',
   '../Results/ED_results/double_well_tests/results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_q_H47f.txt',
    ]

ED_q2 = [
    '../Results/ED_results/double_well_tests/results/model_H47b_rank_4_basis_400_time_200/ACF_ABS_CC_q2_H47b.txt',
    '../Results/ED_results/double_well_tests/results/model_H47c_rank_4_basis_400_time_200/ACF_ABS_CC_q2_H47c.txt',
    '../Results/ED_results/double_well_tests/results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_q2_H47d.txt',
    '../Results/ED_results/double_well_tests/results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_q2_H47e.txt',
    '../Results/ED_results/double_well_tests/results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_q2_H47f.txt',
    ]

# ed order in b,c,d,e,f
disp = [0,-9,-7,7,-14]

aligned_disp = [-7,-14,0,7]
aligned_disp_index = [2,4,0,3] # matches cc idx
# ED 


for idx,cc_path in enumerate(CC_dirs):
    print('displacement:',aligned_disp[idx])
    cc_normed_path = CC_norm_dirs[idx]
    
    cc_energy_path = join(cc_path,CC_filenames[0])
    cc_norm_path = join(cc_path,CC_filenames[1])
    cc_q_path = join(cc_path,CC_filenames[2])
    cc_q2_path = join(cc_path,CC_filenames[3])
        
    ed_energy_path = ED_energy[aligned_disp_index[idx]]
    ed_norm_path = ED_norm[aligned_disp_index[idx]]
    ed_q_path = ED_q[aligned_disp_index[idx]]
    ed_q2_path = ED_q2[aligned_disp_index[idx]]

    cc_energy_data = np.loadtxt(cc_energy_path,skiprows=1)
    cc_norm_data = np.loadtxt(cc_norm_path,skiprows=1)
    cc_q_data = np.loadtxt(cc_q_path,skiprows=1)
    cc_q2_data = np.loadtxt(cc_q2_path,skiprows=1)
        
    ed_energy_data = np.loadtxt(ed_energy_path,skiprows=1)
    ed_norm_data = np.loadtxt(ed_norm_path,skiprows=1)
    ed_q_data = np.loadtxt(ed_q_path,skiprows=1)
    ed_q2_data = np.loadtxt(ed_q2_path,skiprows=1)
    
    
    # q_corrections
    q_ED = ed_q_data[:,1] + aligned_disp[idx]
    q2_ED = ed_q2_data[:,1] + 2*aligned_disp[idx]*ed_q_data[:,1] + aligned_disp[idx]**2
    
    plt.plot(ed_energy_data[:,0],ed_energy_data[:,1],label='ED',color='C0')
    plt.plot(cc_energy_data[:,0],cc_energy_data[:,1],'--',label='CC',color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\langle\hat{H}\rangle$ eV')
    # plt.legend()
    plt.savefig(f'../Results/ED_results/double_well_tests/plots/Energy_d{aligned_disp[idx]}.svg',)
    plt.close()


    
    plt.plot(ed_norm_data[:,0],ed_norm_data[:,1],label='ED',color='C0')
    plt.plot(cc_norm_data[:,0],cc_norm_data[:,1],'--',label='CC',color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\langle\psi(\tau)|\psi(\tau)\rangle$')
    # plt.legend()
    plt.savefig(f'../Results/ED_results/double_well_tests/plots/norm_d{aligned_disp[idx]}.svg',)
    plt.close()
    
    
    plt.plot(ed_q_data[:,0],q_ED,label='ED',color='C0')
    plt.plot(cc_q_data[:,0],cc_q_data[:,1],'--',label='CC',color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\langle\hat{q}\rangle$')
    # plt.legend()
    plt.savefig(f'../Results/ED_results/double_well_tests/plots/q_d{aligned_disp[idx]}.svg',)
    plt.close()
    
    plt.plot(ed_q2_data[:,0],q2_ED,label='ED',color='C0')
    plt.plot(cc_q2_data[:,0],cc_q2_data[:,1],'--',label='CC',color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\langle\hat{q}^2\rangle$')
    # plt.legend()
    plt.savefig(f'../Results/ED_results/double_well_tests/plots/q2_d{aligned_disp[idx]}.svg',)
    plt.close()
    
    plt.plot(ed_q2_data[:,0],q2_ED-q_ED**2,label='ED',color='C0')
    plt.plot(cc_q2_data[:,0],cc_q2_data[:,1]-cc_q_data[:,1]**2,'--',label='CC',color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\text{Var}(\hat{q})$')
    # plt.legend()
    plt.savefig(f'../Results/ED_results/double_well_tests/plots/var_d{aligned_disp[idx]}.svg',)
    plt.close()

    

    
