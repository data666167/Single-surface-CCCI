from signal_processing import excite_extraction as extr
import sys
import os
from os.path import abspath, join, dirname, basename
import numpy as np
import matplotlib.pyplot as plt

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
    paths = ['../Results/CC_results/CH3CN_trial_1000/results/model_ch3cn_ev_rank_4_Z_4_time_1000_dist_0.5/ACF_ABS_CC_ch3cn_ev_h4_Z4_T1000.0_normalized.txt',
             '../Results/CC_results/CH3CN_trial_1000/results/model_ch3cn_ev_rank_4_Z_4_time_1000_dist_0.5/norm_ch3cn_ev_h4_Z4_T1000.0_normalized.txt',
             '../Results/CC_results/CH3CN_trial_1000/results/model_ch3cn_ev_rank_4_Z_4_time_1000_dist_0.5/exp_H_ch3cn_ev_h4_Z4_T1000.0_normalized.txt']
    evals  = evals_ch3cn
    output_path = '../Results/CC_results/CH3CN_trial_1000/tabulations'
    
if dataset == 'CH2NH':
    paths = ['../Results/CC_results/CH2NH_trial_1000/results/model_ch2nh_evp_rank_4_Z_4_time_1000_dist_0.5/ACF_ABS_CC_ch2nh_evp_h4_Z4_T1000.0_normalized.txt',
             '../Results/CC_results/CH2NH_trial_1000/results/model_ch2nh_evp_rank_4_Z_4_time_1000_dist_0.5/norm_ch2nh_evp_h4_Z4_T1000.0_normalized.txt',
             '../Results/CC_results/CH2NH_trial_1000/results/model_ch2nh_evp_rank_4_Z_4_time_1000_dist_0.5/exp_H_ch2nh_evp_h4_Z4_T1000.0_normalized.txt']
    evals = evals_ch2nh
    output_path = '../Results/CC_results/CH2NH_trial_1000/tabulations'
    
plot_extracted = True 

times = [200,500,1000]
for time in times:
    df ,stick_spec ,spec = extr.extract_freqeuncy(paths[0],time,evals,prominence=(0.002))
    file_name = os.path.splitext(basename(paths[0]))[0]+f'_t{time}.csv'
    output_file_path = join(output_path,file_name) 

    
    if plot_extracted == True:
        plt.plot(spec[0],np.abs(spec[1]))
        plt.vlines(stick_spec[0],0,stick_spec[1],color="C3", linewidth=1.0)
        plt.show()


# plotting
## plot norm 
plot_time,norm_time = extr.read_corr_func(paths[1])

plt.plot(plot_time,norm_time)
plt.xlabel('time (fs)')
plt.ylabel(r'Norm $\langle\psi(\tau)|\psi(\tau)\rangle$')
plt.show()

## plot energy choose a scale large enough to see decimal ev changes 
plot_time,energy_time = extr.read_corr_func(paths[2]) 


# inital energy 
print('initial energy (cm):',energy_time[0])

plt.plot(plot_time,energy_time)
plt.xlabel('time (fs)')
plt.ylabel(r'Energy $\langle\hat{H}(\tau)\rangle$ eV')
plt.show()
