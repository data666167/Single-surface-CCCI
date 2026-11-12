import sys
import os
from os.path import abspath, join, dirname, basename
import numpy as np
import pandas as pd
import re
from scipy.optimize import linear_sum_assignment
# ev to cm conversion factor 
conv = 8065.54

Z = [2,4]
for Z_order in Z:
    dataset = 'CH2NH'
    if dataset == 'CH3CN':
        csv_paths = [
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d5M1_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d5M5_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d5M6_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M2_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M3_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M4_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M7_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M8_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M9_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M10_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M11_normalized.csv',
                    f'../Results/CC_results/CH3CN_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch3cn_ev_h4_Z{Z_order}_T500.0_d15M12_normalized.csv',
                    ]
        output_path = f'../Results/CC_results/ch3cn_tabulated_results/Z{Z_order}'
        number_of_modes = 12
        number_of_modes += 1 

    if dataset == 'CH2NH':
        csv_paths = [
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d5M1_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d5M2_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d5M3_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M4_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M5_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M6_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M7_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M8_normalized.csv',
            f'../Results/CC_results/CH2NH_single_mode/tabulations/Z{Z_order}/ACF_ABS_CC_ch2nh_evp_h4_Z{Z_order}_T500.0_d1M9_normalized.csv',
            ]
        output_path = f'../Results/CC_results/ch2nh_tabulated_results/Z{Z_order}'
        number_of_modes = 9
        number_of_modes += 1



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
        df.loc[1:, 'nearest extracted eigenvalues'] += -ex_ref # using mctdh reference  
        # df.loc[1:, 'nearest extracted eigenvalues'] += -average_gs # using avg single reference
        # reference avg groundstate avg comparsion for excitations
    
    
        # ev to wavenumber
        df['nearest extracted eigenvalues'] *= conv
        comp_dict[os.path.splitext(basename(csv_path))[0]] = df['nearest extracted eigenvalues']

    comp_df = pd.DataFrame(comp_dict)    
    
    # value mode assignment
    labeled_mode_cols = []
    for column_name in comp_df.columns:
        if column_name == 'reference eigenvalues':
            continue
        else:
            values = comp_df[column_name].round(0).astype('Int64').astype("string")
            mode_index = re.search(r'M(\d+)_',column_name).group(1)
            mode_expressions = values+"$^{"+mode_index+"}$"
            comp_df["mode "+mode_index] = mode_expressions
            labeled_mode_cols.append("mode "+mode_index)
            # labels_for_mode_cols.append(mode_index)

    comp_df["all modes values"] = comp_df[labeled_mode_cols].apply(lambda row: row.str.cat(sep=","),axis=1,)
    # column mode excitation assignment 


    mode_excitation_count_cols = []
    for i in range(1,number_of_modes):
        mode_excitation_count_cols.append(f'mode {i} count')
        nonempty = comp_df[f"mode {i}"].notna()
        counts = (nonempty.cumsum()-1).astype("Int64").astype("string")
        comp_df[f"mode {i} count"] = (str(i) + "$^{" + counts + "}$").where(nonempty)
    
    comp_df["all modes counts"] = comp_df[mode_excitation_count_cols].apply(lambda row: row.str.cat(sep=","),axis=1,)

    comp_df.to_csv(output_path+'.csv', index=False)
    
    # Excitations with the average
    average_gs = 0
    for csv_path in csv_paths:
       df_avg = pd.read_csv(csv_path)
       average_gs  += df_avg.loc[0,'nearest extracted eigenvalues'] 
    average_gs /= len(csv_paths)
    
    print('molecular_system:',dataset,'Z:',Z_order,'gs_mctdh:',ex_ref*conv,'avg gs',average_gs*conv)
    
    # create new df where excitations are mapped to excitation mean 
    mctdh_excitations = df_ref['reference eigenvalues'].iloc[1:]
    mctdh_excitations = mctdh_excitations.to_numpy()
    excitations_dict = {'reference excitations': mctdh_excitations}
    for csv_path in csv_paths:
        # excitations relative reference groundstate
        df = pd.read_csv(csv_path)
        # df.loc[1:, 'nearest extracted eigenvalues'] += -ex_ref # using mctdh reference  
        df.loc[1:, 'nearest extracted eigenvalues'] += -average_gs # using avg single reference
        # reference avg groundstate avg comparsion for excitations
        excitations = df['nearest extracted eigenvalues'].iloc[1:]
        excitations = excitations.dropna().to_numpy(dtype=float)*conv
        
        organized = np.full(mctdh_excitations.shape, np.nan, dtype=float)
        
        distances = np.abs(mctdh_excitations[:,None]-excitations[None,:])
        ref_index,excite_index = linear_sum_assignment(distances)
        organized[ref_index] = excitations[excite_index] 
        
        excitations_dict[os.path.splitext(basename(csv_path))[0]] = organized
        # breakpoint()
    
    excitations_df = pd.DataFrame(excitations_dict)
    # value mode assignment
    labeled_mode_cols = []
    for column_name in excitations_df.columns:
        if column_name == 'reference excitations':
            continue
        else:
            values = excitations_df[column_name].round(0).astype('Int64').astype("string")
            mode_index = re.search(r'M(\d+)_',column_name).group(1)
            mode_expressions = values+"$^{"+mode_index+"}$"
            excitations_df["mode "+mode_index] = mode_expressions
            labeled_mode_cols.append("mode "+mode_index)
            # labels_for_mode_cols.append(mode_index)

    excitations_df["all modes values"] = excitations_df[labeled_mode_cols].apply(lambda row: row.str.cat(sep=","),axis=1,)
    
    mode_excitation_count_cols = []
    for i in range(1,number_of_modes):
        mode_excitation_count_cols.append(f'mode {i} count')
        nonempty = excitations_df[f"mode {i}"].notna()
        counts = (nonempty.cumsum()).astype("Int64").astype("string")
        excitations_df[f"mode {i} count"] = (str(i) + "$^{" + counts + "}$").where(nonempty)
    
    excitations_df["all modes counts"] = excitations_df[mode_excitation_count_cols].apply(lambda row: row.str.cat(sep=","),axis=1,)

    excitations_df.to_csv(output_path+'_mean.csv', index=False)

