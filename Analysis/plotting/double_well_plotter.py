import matplotlib.pyplot as plt
import numpy as np

# plot spectra case 1,2,3,4 
ED_paths =['../Results/ED_results/double_well_tests/spectra/case_1_90_ED_H47d.pl',
'../Results/ED_results/double_well_tests/spectra/case_2_90_ED_H47e.pl',
'../Results/ED_results/double_well_tests/spectra/case_3_10_ED_H47f.pl',
'../Results/ED_results/double_well_tests/spectra/case_4_10_ED_H47b.pl',]

CC_paths =['../Results/CC_results/double_well_tests/spectra/case_1_90_CC_H47b.pl',
'../Results/CC_results/double_well_tests/spectra/case_2_90_CC_H47b.pl',
'../Results/CC_results/double_well_tests/spectra/case_3_10_CC_H47b.pl',
'../Results/CC_results/double_well_tests/spectra/case_4_10_CC_H47b.pl',]

# Exact eigenvalues

exact_evals = np.loadtxt('../Results/ED_results/double_well_tests/results/model_H47b_rank_4_basis_400_time_200/eigH47b',skiprows=1)

# plot spectra
for idx in range(4):
    print(f'case: {idx+1}')
    spectra_ED_data = np.loadtxt(ED_paths[idx],skiprows=4)
    freq_ED = spectra_ED_data[:,0]
    spec_ED = spectra_ED_data[:,1]

    spectra_CC_data = np.loadtxt(CC_paths[idx],skiprows=4)
    freq_CC = spectra_CC_data[:,0]
    spec_CC = spectra_CC_data[:,1]
    
    if idx < 2:
        plt.vlines(exact_evals,0,np.max(spec_ED)*1.1,colors='red',linestyles=':')
    
    plt.plot(freq_ED, spec_ED, label='ED spectra')
    plt.plot(freq_CC, np.abs(spec_CC), label='CC spectra', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

# plot ACF case 3 
ED_path_case_3 = '../Results/ED_results/double_well_tests/results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_H47f.txt'
CC_path_case_3 = '../Results/CC_results/double_well_tests/results/model_H47b_rank_4_Z_4_time_200_dist_-14/ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt'

acf_ED_data = np.loadtxt(ED_path_case_3,skiprows=1)
acf_CC_data = np.loadtxt(CC_path_case_3,skiprows=1)

acf_time_CC = acf_CC_data[:,0]
acf_CC = acf_CC_data[:,1]

acf_time_ED = acf_ED_data[:,0]
acf_ED = acf_ED_data[:,1]

plt.plot(acf_time_ED, acf_ED, label='ED spectra')
plt.plot(acf_time_CC, acf_CC, label='CC spectra', ls='--')
plt.legend()
plt.xlim(0,100)
plt.tight_layout()
plt.show()
