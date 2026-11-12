import matplotlib.pyplot as plt 
import numpy as np 

CC_paths =[
    '../Results/CC_results/reproduce_cc_for_mctdh/spec/ch3cn_90_d.5.pl',
    '../Results/CC_results/reproduce_cc_for_mctdh/spec/ch3cn_90_d.7071.pl',
    '../Results/CC_results/reproduce_cc_for_mctdh/spec/ch2nh_90_d.5.pl',
    '../Results/CC_results/reproduce_cc_for_mctdh/spec/ch2nh_90_d.7071.pl',
]
MCTDH_paths =[
    '../Results/MCTDH_results/spec/ch3cn_90_d.5.pl',
    '../Results/MCTDH_results/spec/ch3cn_90_d.7071.pl',
    '../Results/MCTDH_results/spec/ch2nh_90_d.5.pl',
    '../Results/MCTDH_results/spec/ch2nh_90_d.7071.pl',
]  

for idx, cc_path in enumerate(CC_paths):
    spectra_CC_data = np.loadtxt(cc_path,skiprows=4)
    spectra_MCTDH_data = np.loadtxt(MCTDH_paths[idx],skiprows=4)
    
    freq_CC = spectra_CC_data[:,0]
    spec_CC = spectra_CC_data[:,1]
    freq_MCTDH = spectra_MCTDH_data[:,0]
    spec_MCTDH = spectra_MCTDH_data[:,1]
    
    plt.plot(freq_MCTDH,np.abs(spec_MCTDH))
    plt.plot(freq_CC,np.abs(spec_CC),'--')
    plt.show()


# investigations of CC efficacy create tables
normal_CC_path = '../Results/CC_results/reproduce_cc_for_mctdh/spec/ch3cn_90_d.7071.pl'
T_1_only_path = '../Results/CC_results/CH3CN_study/spec/ch3cn_90_d.7071_only_T1.pl'
no_T_1_path = '../Results/CC_results/CH3CN_study/spec/ch3cn_90_d.7071_no_T1.pl'
normal_CC_z2_path = '../Results/CC_results/CH3CN_study/spec/ch3cn_90_d.7071_Z2.pl'


spectra_CC_normal_data = np.loadtxt(normal_CC_path,skiprows=4)
spectra_T_1_only_data = np.loadtxt(T_1_only_path,skiprows=4)
spectra_no_T_1_data = np.loadtxt(no_T_1_path,skiprows=4)
spectra_CC_normal_z2_data = np.loadtxt(normal_CC_z2_path,skiprows=4)

plt.plot(spectra_CC_normal_data[:,0],spectra_CC_normal_data[:,1])
plt.plot(spectra_T_1_only_data[:,0],spectra_T_1_only_data[:,1],'--b')
plt.show()

plt.plot(spectra_CC_normal_data[:,0],spectra_CC_normal_data[:,1])
plt.plot(spectra_CC_normal_z2_data[:,0],spectra_CC_normal_z2_data[:,1],':g')
plt.show()

plt.plot(spectra_CC_normal_data[:,0],spectra_CC_normal_data[:,1],'C1')
plt.plot(spectra_no_T_1_data[:,0],spectra_no_T_1_data[:,1],'C0')
plt.ylabel(r'$|S(\omega)|$(arb.units)')
plt.xlabel(r'$\omega$(eV)')
plt.show()


energy_cc = ['../Results/CC_results/CH3CN_trial_1000/results/model_ch3cn_ev_rank_4_Z_4_time_1000_dist_0.5/exp_H_ch3cn_ev_h4_Z4_T1000.0_normalized.txt',
'../Results/CC_results/CH2NH_trial_1000/results/model_ch2nh_evp_rank_4_Z_4_time_1000_dist_0.5/exp_H_ch2nh_evp_h4_Z4_T1000.0_normalized.txt']

for idx in range(2):
    energy_CC_data = np.loadtxt(energy_cc[idx],skiprows=1)
    time_CC = energy_CC_data[:,0]
    energy_CC_corr = energy_CC_data[:,1]
    breakpoint()
    plt.plot(time_CC,energy_CC_corr,color='C1')
    plt.xlabel('time (fs)')
    plt.ylabel(r'$\langle\hat{H}\rangle$ eV')
    plt.show()