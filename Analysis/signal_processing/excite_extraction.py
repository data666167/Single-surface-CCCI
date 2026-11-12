import numpy as np 
import matplotlib.pyplot as plt
# import signal_process as sp
from . import signal_process as sp
import scipy as scp 
import pandas as pd
import os as od

# consistent hbar 
hbar = 0.6582119569

def nearest_indices(ref, x):
    # ref: shape (N,)
    # x: shape (M,)
    # Returns indices of ref with minimal distance to each x[i]
    # shape (M,)
    diff = np.abs(ref[:, None] - x[None, :])  # shape (N, M)
    idx = diff.argmin(axis=0)                 # shape (M,)
    return idx

def build_nearest_table(ref, arrays):
    """
    ref: 1D numpy array of shape (N,)
    arrays: list of 1D numpy arrays, each length <= N

    Returns: 2D array of shape (N, 1 + len(arrays))
             column 0: ref
             column j: values from arrays[j-1] placed at nearest ref row, NaN elsewhere
    """
    ref = np.asarray(ref)
    N = ref.shape[0]
    k = len(arrays)

    # Initialize result
    table = np.full((N, 1 + k), np.nan)
    table[:, 0] = ref

    for j, arr in enumerate(arrays, start=1):
        arr = np.asarray(arr)
        idx = nearest_indices(ref, arr)  # row indices in ref

        # If multiple values map to the same row, last one wins; adjust if you want different behavior
        table[idx, j] = arr

    return table

def extract_freqeuncy(acf_file_path,final_time,ref_eig_vals,acf_stepsize=0.2,freq_range = (1,2.5),height=0.0015,prominence=(0.002)):
    # construct acf 
    cc_data  = np.loadtxt(acf_file_path,skiprows=1)

    cc_time = cc_data[:,0]

    cc_acf =  cc_data[:,1] + 1j*cc_data[:,2]
    
    time_cc_idx = int(final_time//acf_stepsize) + 1 
    prop_cc_time = cc_time[:time_cc_idx]
    prop_cc_acf = cc_acf[:time_cc_idx]*scp.signal.windows.hamming(len(cc_acf[:time_cc_idx]))
    
    freqs = -np.linspace(freq_range[0],freq_range[1],2000) # sign for fourier transform
    cc_spec = []
    for freq in freqs:
        cc_spec.append(((sp.fourier_transforms(prop_cc_acf,(prop_cc_time/hbar),None,freq))))
    # make spec list an array 
    cc_spec = np.array(cc_spec)
    
    # extract peak frequencies 
    peak_locs, props = scp.signal.find_peaks(np.abs(cc_spec),height=height,prominence=prominence)
    # breakpoint()
    
    # found peaks for plotting 
    ## fix sign for plot of fourier transform
    plotting_freqs = -1*freqs
    
    found_peaks_freqs = plotting_freqs[peak_locs] 
    found_peak_intensities = np.abs(cc_spec[peak_locs])
    
    # extract newtons improved peaks
    stick_spec_intensities = []
    ex_opt = []
    for peak_idx in peak_locs:
        opt_freq ,_ = sp.newtons_method_pwr(prop_cc_acf,(prop_cc_time/hbar),freq_guess=freqs[peak_idx],freq_left=freqs[peak_idx+1],freq_right=freqs[peak_idx-1])
        intensity = sp.fourier_transforms(prop_cc_acf,(prop_cc_time/hbar),acf_stepsize,opt_freq)
        stick_spec_intensities.append(np.abs(intensity))
        ex_opt.append(-opt_freq)
    
    # sort extracted freqs for comparsion 
    ex_opt = np.sort(np.array(ex_opt).real)
    ref_eig_vals = np.sort(np.array(ref_eig_vals).real)
    table = build_nearest_table(ref_eig_vals, [ex_opt])
    # breakpoint()
    df_table = pd.DataFrame(table,columns=['reference eigenvalues','nearest extracted eigenvalues'])
    
    return(df_table,[ex_opt,stick_spec_intensities],[plotting_freqs,cc_spec])

def read_corr_func(corr_func_file_path,):
    cc_data = np.loadtxt(corr_func_file_path,skiprows=1)

    cc_time = cc_data[:,0]

    cc_corr_func =  cc_data[:,1] + 1j*cc_data[:,2]
    
    return(cc_time,cc_corr_func)