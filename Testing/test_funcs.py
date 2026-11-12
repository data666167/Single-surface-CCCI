# system imports
import io
import time
import os
import itertools as it
from os.path import abspath, join, dirname, basename
import argparse
import sys
import cProfile
import pstats


# third party imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp 
import pandas as pd

# import the path to the package
project_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import project

# local imports
from project.residual_equations.symmetrize import symmetrize_tensor
from project.vibronic_hamiltonian import vibronic_hamiltonian
from project.vibronic import vIO, VMK, model_op
import project.spectra
import project.log_conf

def symmetrize_upper_triangular(arr):
    """
    Symmetrize an array that is upper triangular in the last k indices.
    The array shape is (A, A, N, N, ..., N) with k N's.
    The symmetrization is done over the last k indices.
    ----------------------------------------------------------------
    THIS has been shown to work for a 3 mode quadratic hamiltonian
    ----------------------------------------------------------------
    """
    shape = arr.shape
    A = shape[0]
    k = len(shape) - 2  # number of N indices
    N = shape[2]
    
    # Generate all permutations of the last k indices
    perms = list(it.permutations(range(k)))
    
    # Initialize symmetrized array
    sym_arr = np.zeros_like(arr, dtype=np.result_type(arr, float))
    
    # Sum over all permutations of the last k indices
    for perm in perms:
        # Construct the axes order for transpose
        axes_order = [0, 1] + [i+2 for i in perm]
        sym_arr += np.transpose(arr, axes=axes_order)
    
    # Average over the number of permutations
    sym_arr /= len(perms)
    return(sym_arr)
    
    
def replace_entire_line_with_keyword(filename, keyword, new_line):
    """Replace entire line that contains a keyword as a whole word using os.system"""
    # Escape special characters for sed
    escaped_new_line = new_line.replace('/', r'\/')
    escaped_keyword = keyword.replace('/', r'\/')
    
    # Use word boundaries to match whole words only
    command = f"sed -i '/\\b{escaped_keyword}\\b/c\\{escaped_new_line}' {filename}"
    return_code = os.system(command)
    return return_code == 0

def replace_last_two_lines(filename, second_last_content, last_content):
    """
    Replace specifically the last 2 lines of a file.
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        if len(lines) < 2:
            print("Error: File must have at least 2 lines")
            return False
        
        # Replace second-to-last line
        lines[-2] = second_last_content + '\n' if not second_last_content.endswith('\n') else second_last_content
        
        # Replace last line
        lines[-1] = last_content + '\n' if not last_content.endswith('\n') else last_content
        
        # Write back
        with open(filename, 'w') as file:
            file.writelines(lines)
        
        print("Last 2 lines replaced successfully")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_model_from_json_file(path, order):
 
    model = vIO.load_model_from_JSON(path)

    A, N = vIO._extract_dimensions_from_dictionary(model)
    

    if mctdh_comp := True:
        # breakpoint()
        # This will handle JSONS which match the coefficients of .op file which are written in the upper triangular format
        # You will need to talk to neil about trade offs for coefficient lengths  
        # breakpoint()
        temp_g2 = model[VMK.G2].reshape(A,A,N,N)
        temp_g2 = symmetrize_upper_triangular(temp_g2)
        model[VMK.G2] = temp_g2.reshape(np.shape(model[VMK.G2]))
        # breakpoint()
        temp_g3 = model[VMK.G3].reshape(A,A,N,N,N)
        temp_g3 = symmetrize_upper_triangular(temp_g3)
        model[VMK.G3] = temp_g3.reshape(np.shape(model[VMK.G3]))
        
        temp_g4 = model[VMK.G4].reshape(A,A,N,N,N,N)
        temp_g4 = symmetrize_upper_triangular(temp_g4)
        model[VMK.G4] = temp_g4.reshape(np.shape(model[VMK.G4]))
                  
    model[VMK.etdm].fill(complex(0.1))
    model[VMK.mtdm].fill(complex(0.1))

    # swap electron / vibrational dimensions
    vIO.prepare_model_for_cc_integration(model, order)
 
    return model

def get_doublesim_model_from_json_file(path, order,t):
    """ This is sets up the coefficients for the alternative
    double similarity transformation that can be calculated for testing purposes"""

    model = vIO.load_model_from_JSON(path)

    A, N = vIO._extract_dimensions_from_dictionary(model)
    
    assert N == 1, 'This test can only be done for single mode systems'
    

    w = model[VMK.w].copy() 
    v_i = model[VMK.G1].copy().flatten()
    v_ii = model[VMK.G2].copy().flatten()
    v_iii = model[VMK.G3].copy().flatten()
    v_iiii = model[VMK.G4].copy().flatten()
    
    t *= 1

    if order >= 1:    
        model[VMK.E] += np.sqrt(2)*t*v_i + w*(t**2) 
        model[VMK.G1][:,:,0] += np.sqrt(2)*w*t

    if order >= 2:    
        model[VMK.E] += (2*(t**2)*v_ii) 
        model[VMK.G1][:,:,0] += (2*np.sqrt(2)*t*v_ii)

    if order >= 3:    
        model[VMK.E] += (2*np.sqrt(2)*(t**3)*v_iii) 
        model[VMK.G1][:,:,0] += (6*(t**2)*v_iii)
        model[VMK.G2][:,:,:,0] += 3*np.sqrt(2)*t*v_iii 

    if order >= 4:    
        model[VMK.E] += (4*(t**4)*v_iiii) 
        model[VMK.G1][:,:,0] += (8*np.sqrt(2)*(t**3)*v_iiii)
        model[VMK.G2][:,:,:,0] += 12*(t**2)*v_iiii
        model[VMK.G3][:,:,:,:,0] += 4*np.sqrt(2)*t*v_iiii 
    
    # breakpoint()
    number_of_modes = 1
    surfaces = 1
    init_E = 0

    etdm = np.zeros((1,surfaces))+1
    mtdm = np.zeros((1,surfaces))+1
    E = np.zeros((surfaces,surfaces))
    w = np.zeros((number_of_modes))


    test_dict = {VMK.N: number_of_modes ,
                    VMK.A: surfaces,
                    VMK.E: E,
                    VMK.w: w,
                    VMK.etdm: etdm,
                    VMK.mtdm: mtdm,
                    }
    
    test_dict[VMK.E ] = model[VMK.E]
    test_dict[VMK.w] = model[VMK.w]
    test_dict[VMK.G1] = model[VMK.G1] 
    test_dict[VMK.G2] = model[VMK.G2] 
    test_dict[VMK.G3] = model[VMK.G3] 
    test_dict[VMK.G4] = model[VMK.G4] 
        
    # the harmonic oscillator frequencies are unchanged after double transform 
    # fourth order coeffeicents are unchanged after double transform  
    # model[VMK.w] = model[VMK.w] 
    # model[VMK.G4] = model[VMK.G4]
    
    model[VMK.etdm].fill(complex(0.1))
    model[VMK.mtdm].fill(complex(0.1))

    # swap electron / vibrational dimensions
    vIO.prepare_model_for_cc_integration(model, order)
    
    return model

def get_singlesim_model_from_json_file(path, order,t):
    """ This is sets up the coefficients for the alternative
    double similarity transformation that can be calculated for testing purposes"""

    model = vIO.load_model_from_JSON(path)

    A, N = vIO._extract_dimensions_from_dictionary(model)
    
    assert N == 1, 'This test can only be done for single mode systems'
    

    w = model[VMK.w].copy() 
    v_i = model[VMK.G1].copy().flatten()
    v_ii = model[VMK.G2].copy().flatten()
    v_iii = model[VMK.G3].copy().flatten()
    v_iiii = model[VMK.G4].copy().flatten()
    
     
    if order >= 1:    
        model[VMK.E] += (np.sqrt(2)/2)*t*v_i + 0.25*w*(t**2) 
        model[VMK.G1][:,:,0] += 0.5*np.sqrt(2)*w*t

    if order >= 2:    
        model[VMK.E] += 0.5*((t**2)*v_ii) 
        model[VMK.G1][:,:,0] += (np.sqrt(2)*t*v_ii)


    if order >= 3:    
        model[VMK.E] += (np.sqrt(2)/4)*((t**3)*v_iii) 
        model[VMK.G1][:,:,0] += (1.5*(t**2)*v_iii)
        model[VMK.G2][:,:,:,0] += (3*np.sqrt(2)/2)*t*v_iii 

    if order >= 4:    
        model[VMK.E] += 0.25*((t**4)*v_iiii) 
        model[VMK.G1][:,:,0] += (np.sqrt(2)*(t**3)*v_iiii)
        model[VMK.G2][:,:,:,0] += 3*(t**2)*v_iiii
        model[VMK.G3][:,:,:,:,0] += 2*np.sqrt(2)*t*v_iiii
        
    # the harmonic oscillator frequencies are unchanged after double transform 
    # fourth order coeffeicents are unchanged after double transform  
    # model[VMK.w] = model[VMK.w] 
    # model[VMK.G4] = model[VMK.G4]
    
    model[VMK.etdm].fill(complex(0.1))
    model[VMK.mtdm].fill(complex(0.1))

    # swap electron / vibrational dimensions
    vIO.prepare_model_for_cc_integration(model, order)
    
    return model

def double_similarity_test(model, alt_double_model,alt_single_model ,file_name, order, t_1,output_path):
    """ alt model comes from the double sim json reader"""

    # initialize objects 
    # truncation order of Z and T do not affect the initialization they are set 1
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=1,
        calculate_population_flag=False,
    )
    
    alt_single_trans_ham = vibronic_hamiltonian(
        alt_single_model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=1,
        calculate_population_flag=False,
    )
    
    # truncation order of Z and T do not affect the initialization they are set 1
    alt_double_trans_ham = vibronic_hamiltonian(
        alt_double_model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=1,
        calculate_population_flag=False,
    )


    t_dict ={1:t_1}
    # perform double similarity transform on regular model
    single_trans_ham = hamiltonian._similarity_trans_Oz(hamiltonian.h,t_dict)
    double_trans_ham = hamiltonian._cal_H_bar_tilde_Oz(single_trans_ham,t_dict)
    
     
    # write data to file 
    print([[*i] for i in [*single_trans_ham]])
    e = np.array([m for l,m in enumerate([k for j in [[*i] for i in [*single_trans_ham]] for k in j]) if (l+1)%2 == 1])
    f = np.array([m for l,m in enumerate([k for j in [[*i] for i in [*single_trans_ham]] for k in j]) if (l+1)%2 == 0])
    a = np.array([i.flatten() for i in [*single_trans_ham.values()]]).flatten()
    b = np.array([i.flatten() for i in [*alt_single_trans_ham.h.values()]]).flatten()
    c = np.array([i.flatten() for i in [*double_trans_ham.values()]]).flatten()
    d = np.array([i.flatten() for i in [*alt_double_trans_ham.h.values()]]).flatten()
    arr=np.stack((e,f,a,b,c,d)).T.real
    
    spacing = 10
    int_spacing = 4 
    np.savetxt(output_path, arr, 
           header=f'{"H^n":>{int_spacing}} {"H_m":>{int_spacing}} {"single_trans":>{spacing}} {"alt_single_trans":>{spacing}} {"double_trans":>{spacing}} {"alt_double_trans":>{spacing}}',
           fmt=f'%{int_spacing}d %{int_spacing}d %{spacing}f %{spacing}f %{spacing}f %{spacing}f')
    
    return


 

def upper_tri_nd(shape):
    """Generate upper-triangular index tuples for any N-D shape."""
    for idx in np.ndindex(shape):
        if list(idx) == sorted(idx):  # ensures i0 <= i1 <= i2 ...
            yield idx


def permutations_with_duplicates_count(t):
    import math
    from collections import Counter
    n = len(t)
    freq = Counter(t)
    denom = 1
    for c in freq.values():
        denom *= math.factorial(c)
    return math.factorial(n) // denom



def single_mode_hamiltonian_construction_residual_vec(model,file_name,order,nZ,vec):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    
    A = hamiltonian.A
    N = hamiltonian.N
    assert N == 1, 'Use for single mode systems only'
    
    Z_keys = [(0,0),(1,0),(2,0),(3,0),(4,0)] 
    Z_shapes = [A,[A,N],[A, N, N],[A, N, N, N],[A, N, N, N, N]]
    Z = {}
    
    for i in range(nZ+1):
        Z[Z_keys[i]] = np.zeros(Z_shapes[i])
    
    T = {
        0: np.zeros(A, dtype=complex),
        1: np.zeros([N], dtype=complex),
    }
    
    T_derivatives = {
            'dT_0': T[0],
            'dT_1': T[1]    
    }
    
    # Define the initial hamiltonian that we reconstructive element by element
    original_hamiltonian = hamiltonian.h
    for key in original_hamiltonian.keys():
        original_hamiltonian[key] *= 1j
    
    reconstructed_hamiltonian = np.zeros((nZ+1,nZ+1)) 
    # mapping is 1 to 1 between keys and CI basis elements  
    for zdx, key_z in enumerate(Z.keys()):
        Z[key_z] += np.sqrt(scp.special.factorial(zdx))*vec[zdx]
    
    # for key in Z.keys():
    #     Z[key] = symmetrize_tensor(1,Z[key],key[0])
        
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    R_z_dict = {}
    for rdx,key_r in enumerate(R.keys()):
        if rdx > 1:
            R_z_dict[key_r] =R[key_r]
    # R = 
    # hamiltonian.derivatives
    # R = hamiltonian.residual_dict
        
    residual_vec = np.zeros(nZ+1)
    for rdx,key_r in enumerate(R_z_dict.keys()):

        if rdx <= nZ:
            # breakpoint()
            residual_vec[rdx] += (1/np.sqrt(scp.special.factorial(rdx)))*R_z_dict[key_r].item()
        else:
            break
    
    return(residual_vec)

def single_mode_hamiltonian_construction_squared_residual_vec(model,file_name,order,nZ,vec):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    
    A = hamiltonian.A
    N = hamiltonian.N
    assert N == 1, 'Use for single mode systems only'
    
    Z_keys = [(0,0),(1,0),(2,0),(3,0),(4,0)] 
    Z_shapes = [A,[A,N],[A, N, N],[A, N, N, N],[A, N, N, N, N]]
    Z = {}
    
    for i in range(nZ+1):
        Z[Z_keys[i]] = np.zeros(Z_shapes[i])
    
    T = {
        0: np.zeros(A, dtype=complex),
        1: np.zeros([N], dtype=complex),
    }
    
    T_derivatives = {
            'dT_0': T[0],
            'dT_1': T[1]    
    }
    
    # Define the initial hamiltonian that we reconstructive element by element
    original_hamiltonian = hamiltonian.h
    for key in original_hamiltonian.keys():
        original_hamiltonian[key] *= 1j
    
    reconstructed_hamiltonian = np.zeros((nZ+1,nZ+1)) 
    # mapping is 1 to 1 between keys and CI basis elements  
    for zdx, key_z in enumerate(Z.keys()):
        Z[key_z] += np.sqrt(scp.special.factorial(zdx))*vec[zdx]
    
        
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    
    Z[(0,0)] = R['dZ_0'].copy()
    Z[(1,0)] = R['dZ_1'].copy()
    Z[(2,0)] = R['dZ_2'].copy()
    if nZ == 4:  
        Z[(3,0)] = R['dZ_3'].copy()
        Z[(4,0)] = R['dZ_4'].copy()
    
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    
    # change Residual keys to be compatible with rest of code  
    r = {}
    r[0] = R['dZ_0'].copy()
    r[1] = R['dZ_1'].copy()
    r[2] = R['dZ_2'].copy()
    if nZ == 4:  
        r[3] = R['dZ_3'].copy()
        r[4] = R['dZ_4'].copy() 
    
    R_z_dict = {}
    for rdx,key_r in enumerate(r.keys()):
        R_z_dict[key_r] =r[key_r]
    
    residual_vec = np.zeros(nZ+1)
    for rdx,key_r in enumerate(R_z_dict.keys()):

        if rdx <= nZ:
            residual_vec[rdx] += (1/np.sqrt(scp.special.factorial(rdx)))*R_z_dict[key_r].item()
        else:
            break
    
    return(residual_vec)

def single_mode_hamiltonian_construction_ci(model,file_name,order,nZ):
    # assert that hamiltonian has 1 mode
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )

    assert hamiltonian.N == 1, 'Use for single mode systems only'  

    # for an nZ of 4 we directly construct the hamiltonian in the CI basis of |0> ... |4> 
    # i.e for a 1 mode system we directly build the hamiltonian in correct basis 
    # and in a logical ordering 
    exact_full_ci_hamiltonian = np.real(hamiltonian._single_mode_FCI_exact_standard_hamiltonian(nZ+1))
    
    return(exact_full_ci_hamiltonian)

def double_mode_hamiltonian_residual(model,file_name,order,nZ):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    
    A = hamiltonian.A
    N = hamiltonian.N
    assert N == 2, 'must be a 2 mode system'
    
    Z_keys = [(0,0),(1,0),(2,0),(3,0),(4,0)] 
    Z_shapes = [A,[A,N],[A, N, N],[A, N, N, N],[A, N, N, N, N]]
    Z = {}
    
    for i in range(nZ+1):
        Z[Z_keys[i]] = np.zeros(Z_shapes[i]) 
    
    T = {
        0: np.zeros(A, dtype=complex),
        1: np.zeros([N], dtype=complex),
    }
    
    T_derivatives = {
            'dT_0': T[0],
            'dT_1': T[1]    
    }
    
    # determine upper triangular Z_indexes
    upper_triangular_z_indexes = []  
    for key_z in Z.keys():
        upper_triangular_z_indexes.extend(upper_tri_nd(Z[key_z].shape))
    
    # map Z-amplitude to product states
    ci_state_map_z_index = []
    for i in upper_triangular_z_indexes:
        i = i[1:] # remove first dimension 
        mode_0 = i.count(0)
        mode_1 = i.count(1)
        ci_state_map_z_index.append((mode_0,mode_1))
    
    # breakpoint()
    # symmetrize_tensor
    
    # Reorder 
    ci_ranked_indexes = []
    for state in ci_state_map_z_index:
        flat_index = (nZ+1)*(state[0]) + state[1]
        ci_ranked_indexes.append(flat_index)   
    
    permutations = np.argsort(ci_ranked_indexes)
    ordered_ci_state_map_z_index = np.array(ci_state_map_z_index)[permutations]
    ordered_ci_state_map_z_index_list = [tuple(int(x)for x in row) 
                                         for row in ordered_ci_state_map_z_index]
    ordered_upper_triangular_z_indexes = [upper_triangular_z_indexes[int(i)] 
                                          for i in permutations]
        
    original_hamiltonian = hamiltonian.h
    for key in original_hamiltonian.keys():
        original_hamiltonian[key] *= 1j
    
    reconstructed_hamiltonian = np.zeros((len(upper_triangular_z_indexes)
                                          ,len(upper_triangular_z_indexes)),
                                         dtype=complex)
    
    for zdx, z_basis_index in enumerate(ordered_upper_triangular_z_indexes):
        
        symmetric_indexes_z = [i for i in set(it.permutations(z_basis_index[1:]))]
        
        for sym_index_z in symmetric_indexes_z:
            true_z_state_index = (0,)+sym_index_z
            if len(z_basis_index) == 1:
                Z[(0,0)][true_z_state_index] += 1
            elif len(z_basis_index) == 2:
                Z[(1,0)][true_z_state_index] += 1
            elif len(z_basis_index) == 3:
                Z[(2,0)][true_z_state_index] += 1
            elif len(z_basis_index) == 4:
                Z[(3,0)][true_z_state_index] += 1
            elif len(z_basis_index) == 5:
                Z[(4,0)][true_z_state_index] += 1
            
        R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
        # uses the neil symmetrized derivatives
        # R = hamiltonian.residual_dict.copy()
        
        residual_vec = np.zeros(len(upper_triangular_z_indexes),dtype=complex)
        for rdx,z_basis_index_r in enumerate(ordered_upper_triangular_z_indexes):
            symmetric_indexes_r = [i for i in set(it.permutations(z_basis_index_r[1:]))]
            for sym_index_r in symmetric_indexes_r:
                true_r_state_index = (0,) + sym_index_r
                if len(z_basis_index_r) == 1:
                    residual_vec[rdx] += R['dZ_0'][true_r_state_index] 
                elif len(z_basis_index_r) == 2:
                    residual_vec[rdx] += R['dZ_1'][true_r_state_index] 
                elif len(z_basis_index_r) == 3:
                    residual_vec[rdx] += 0.5*R['dZ_2'][true_r_state_index] 
                elif len(z_basis_index_r) == 4:
                    residual_vec[rdx] += (1/6)*R['dZ_3'][true_r_state_index] 
                elif len(z_basis_index_r) == 5:
                    residual_vec[rdx] += (1/24)*R['dZ_4'][true_r_state_index] 
               
        
        if symmetrization_test:= True:
            # R['dZ_4'] is already symmetrized we subsume the defintion of R to include the factorial coeff
            # thus when we apply it a second time we must multiply R by the coeff explicitly
            # breakpoint()
            if nZ == 4:    
                assert (np.abs((1/24)*symmetrize_tensor(2,(1/24)*R['dZ_4'],4)-(1/24)*R['dZ_4'])<1e-14).all(),'symmetrization is incorrectly done'
                assert (np.abs((1/6)*symmetrize_tensor(2,(1/6)*R['dZ_3'],3)-(1/6)*R['dZ_3'])<1e-14).all(),'symmetrization is incorrectly done'
            assert (np.abs((1/2)*symmetrize_tensor(2,(1/2)*R['dZ_2'],2)-(1/2)*R['dZ_2'])<1e-14).all(),'symmetrization is incorrectly done' 
        
        
        reconstructed_hamiltonian[:,zdx] =  residual_vec
        # breakpoint()
    
        for key_z in Z.keys():
            Z[key_z] *= 0  
    
    mode_0_coeffs = []
    mode_1_coeffs = [] 
    
    for state in ordered_ci_state_map_z_index_list:
        mode_0_coeffs.append(state[0])
        mode_1_coeffs.append(state[1])
    
    mode_0_coeffs = np.sqrt(scp.special.factorial(np.array(mode_0_coeffs)))
    mode_1_coeffs = np.sqrt(scp.special.factorial(np.array(mode_1_coeffs)))
    normalization_vec = mode_1_coeffs*mode_0_coeffs
    
    normalization_mask = np.outer(normalization_vec,normalization_vec)
    
    reconstructed_hamiltonian *= normalization_mask
    reconstructed_hamiltonian = np.real(reconstructed_hamiltonian)
    
    return(reconstructed_hamiltonian)


def double_mode_hamiltonian_residual_squared_vec(model,file_name,order,nZ,vec):
    hamiltonian = vibronic_hamiltonian(
            model, file_name,
            hamiltonian_truncation_order=order, cc_truncation_order=order,
            T_truncation_order=1, Z_truncation_order=nZ,
            calculate_population_flag=False,
        )
    
    A = hamiltonian.A
    N = hamiltonian.N
    assert N == 2, 'must be a 2 mode system'
    
    Z_keys = [(0,0),(1,0),(2,0),(3,0),(4,0)] 
    Z_shapes = [A,[A,N],[A, N, N],[A, N, N, N],[A, N, N, N, N]]
    Z = {}
    
    for i in range(nZ+1):
        Z[Z_keys[i]] = np.zeros(Z_shapes[i]) 
    
    T = {
        0: np.zeros(A, dtype=complex),
        1: np.zeros([N], dtype=complex),
    }
    
    T_derivatives = {
            'dT_0': T[0],
            'dT_1': T[1]    
    }
    
    # determine upper triangular Z_indexes
    upper_triangular_z_indexes = []  
    for key_z in Z.keys():
        upper_triangular_z_indexes.extend(upper_tri_nd(Z[key_z].shape))
    
    # map Z-amplitude to product states
    ci_state_map_z_index = []
    for i in upper_triangular_z_indexes:
        i = i[1:] # remove first dimension 
        mode_0 = i.count(0)
        mode_1 = i.count(1)
        ci_state_map_z_index.append((mode_0,mode_1))
    
    # Reorder 
    ci_ranked_indexes = []
    for state in ci_state_map_z_index:
        flat_index = (nZ+1)*(state[0]) + state[1]
        ci_ranked_indexes.append(flat_index)   
    
    permutations = np.argsort(ci_ranked_indexes)
    ordered_ci_state_map_z_index = np.array(ci_state_map_z_index)[permutations]
    ordered_ci_state_map_z_index_list = [tuple(int(x)for x in row) 
                                         for row in ordered_ci_state_map_z_index]
    ordered_upper_triangular_z_indexes = [upper_triangular_z_indexes[int(i)] 
                                          for i in permutations]
        
    original_hamiltonian = hamiltonian.h
    for key in original_hamiltonian.keys():
        original_hamiltonian[key] *= 1j
    
    mode_0_coeffs = []
    mode_1_coeffs = [] 
    
    for state in ordered_ci_state_map_z_index_list:
        mode_0_coeffs.append(state[0])
        mode_1_coeffs.append(state[1])
    
    mode_0_coeffs = np.sqrt(scp.special.factorial(np.array(mode_0_coeffs)))
    mode_1_coeffs = np.sqrt(scp.special.factorial(np.array(mode_1_coeffs)))
    normalization_vec = mode_1_coeffs*mode_0_coeffs
    vec = vec*(normalization_vec)
    
    for zdx, z_basis_index in enumerate(ordered_upper_triangular_z_indexes):
            
        symmetric_indexes_z = [i for i in set(it.permutations(z_basis_index[1:]))]
        
        for sym_index_z in symmetric_indexes_z:
            true_z_state_index = (0,)+sym_index_z
            if len(z_basis_index) == 1:
                if sym_index_z == ():
                    sym_index_z = (0,)+sym_index_z
                Z[(0,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 2:
                Z[(1,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 3:
                Z[(2,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 4:
                Z[(3,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 5:
                Z[(4,0)][true_z_state_index] = vec[zdx]
        

    # uses the neil symmetrized derivatives
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    
    Z[(0,0)] = R['dZ_0'].copy()
    Z[(1,0)] = R['dZ_1'].copy()
    Z[(2,0)] = R['dZ_2'].copy()
    if nZ == 4:  
        Z[(3,0)] = R['dZ_3'].copy()
        Z[(4,0)] = R['dZ_4'].copy()
        
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    
    residual_vec = np.zeros(len(upper_triangular_z_indexes),dtype=complex)
    for rdx,z_basis_index_r in enumerate(ordered_upper_triangular_z_indexes):
        if len(z_basis_index_r) == 1:
            residual_vec[rdx] += R['dZ_0'][z_basis_index_r] 
        elif len(z_basis_index_r) == 2:
            residual_vec[rdx] += R['dZ_1'][z_basis_index_r] 
        elif len(z_basis_index_r) == 3:
            residual_vec[rdx] += R['dZ_2'][z_basis_index_r] 
        elif len(z_basis_index_r) == 4:
            residual_vec[rdx] += R['dZ_3'][z_basis_index_r] 
        elif len(z_basis_index_r) == 5:
            residual_vec[rdx] += R['dZ_4'][z_basis_index_r]
    
    normalization = 1/(normalization_vec)
    residual_vec = np.real(residual_vec)
    residual_vec *= normalization
    return(residual_vec)

def double_mode_hamiltonian_residual_vec(model,file_name,order,nZ,vec):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    
    A = hamiltonian.A
    N = hamiltonian.N
    assert N == 2, 'must be a 2 mode system'
    
    Z_keys = [(0,0),(1,0),(2,0),(3,0),(4,0)] 
    Z_shapes = [A,[A,N],[A, N, N],[A, N, N, N],[A, N, N, N, N]]
    Z = {}
    
    for i in range(nZ+1):
        Z[Z_keys[i]] = np.zeros(Z_shapes[i]) 
    
    T = {
        0: np.zeros(A, dtype=complex),
        1: np.zeros([N], dtype=complex),
    }
    
    T_derivatives = {
            'dT_0': T[0],
            'dT_1': T[1]    
    }
    
    # determine upper triangular Z_indexes
    upper_triangular_z_indexes = []  
    for key_z in Z.keys():
        upper_triangular_z_indexes.extend(upper_tri_nd(Z[key_z].shape))
    
    # map Z-amplitude to product states
    ci_state_map_z_index = []
    for i in upper_triangular_z_indexes:
        i = i[1:] # remove first dimension 
        mode_0 = i.count(0)
        mode_1 = i.count(1)
        ci_state_map_z_index.append((mode_0,mode_1))
    
    # Reorder 
    ci_ranked_indexes = []
    for state in ci_state_map_z_index:
        flat_index = (nZ+1)*(state[0]) + state[1]
        ci_ranked_indexes.append(flat_index)   
    
    permutations = np.argsort(ci_ranked_indexes)
    ordered_ci_state_map_z_index = np.array(ci_state_map_z_index)[permutations]
    ordered_ci_state_map_z_index_list = [tuple(int(x)for x in row) 
                                         for row in ordered_ci_state_map_z_index]
    ordered_upper_triangular_z_indexes = [upper_triangular_z_indexes[int(i)] 
                                          for i in permutations]
        
    original_hamiltonian = hamiltonian.h
    for key in original_hamiltonian.keys():
        original_hamiltonian[key] *= 1j
    
    mode_0_coeffs = []
    mode_1_coeffs = [] 
    
    for state in ordered_ci_state_map_z_index_list:
        mode_0_coeffs.append(state[0])
        mode_1_coeffs.append(state[1])
    
    mode_0_coeffs = np.sqrt(scp.special.factorial(np.array(mode_0_coeffs)))
    mode_1_coeffs = np.sqrt(scp.special.factorial(np.array(mode_1_coeffs)))
    normalization_vec = mode_1_coeffs*mode_0_coeffs
    vec = vec*(normalization_vec)
    
    for zdx, z_basis_index in enumerate(ordered_upper_triangular_z_indexes):
            
        symmetric_indexes_z = [i for i in set(it.permutations(z_basis_index[1:]))]
        
        for sym_index_z in symmetric_indexes_z:
            true_z_state_index = (0,)+sym_index_z
            if len(z_basis_index) == 1:
                if sym_index_z == ():
                    sym_index_z = (0,)+sym_index_z
                Z[(0,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 2:
                Z[(1,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 3:
                Z[(2,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 4:
                Z[(3,0)][true_z_state_index] = vec[zdx]
            elif len(z_basis_index) == 5:
                Z[(4,0)][true_z_state_index] = vec[zdx]
        

    # uses the neil symmetrized derivatives
    R = hamiltonian.derivatives_Z_decoupled_scheme(original_hamiltonian,Z,T,T_derivatives)
    
    residual_vec = np.zeros(len(upper_triangular_z_indexes),dtype=complex)
    for rdx,z_basis_index_r in enumerate(ordered_upper_triangular_z_indexes):
        if len(z_basis_index_r) == 1:
            residual_vec[rdx] += R['dZ_0'][z_basis_index_r] 
        elif len(z_basis_index_r) == 2:
            residual_vec[rdx] += R['dZ_1'][z_basis_index_r] 
        elif len(z_basis_index_r) == 3:
            residual_vec[rdx] += R['dZ_2'][z_basis_index_r] 
        elif len(z_basis_index_r) == 4:
            residual_vec[rdx] += R['dZ_3'][z_basis_index_r] 
        elif len(z_basis_index_r) == 5:
            residual_vec[rdx] += R['dZ_4'][z_basis_index_r]
    
    normalization = 1/(normalization_vec)
    residual_vec = np.real(residual_vec)*normalization
    
    
    return(residual_vec)


def exact_FCI_2_mode_quadratic_hamiltonian(model,file_name,order,nZ):
    assert order <=2, 'hamiltonian must be quadratic'
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    hamiltonian._FCI_exact_standard_quad_hamiltonian(10)
    hamiltonian.project_quad_subspace(10)
    exact_hamiltonian = hamiltonian.subspace_hamiltonian
    
    return(exact_hamiltonian)

def exact_FCI_2_mode_hamiltonian(model,file_name,order,nZ):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    hamiltonian._double_mode_FCI_exact_standard_hamiltonian(10)
    hamiltonian.project_subspace(10,)
    exact_hamiltonian = hamiltonian.subspace_hamiltonian
    return(exact_hamiltonian)
 
def residual_check(model,file_name,order,nZ,output_path):
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,
    )
    
    np.random.seed(42)
    
    if hamiltonian.N == 1:
        single_mode_hamiltonian_ED = single_mode_hamiltonian_construction_ci(model, file_name, order, nZ)
        rand_vector = np.random.randint(1,100,size=single_mode_hamiltonian_ED.shape[0])
        
        ed_mat_vec = single_mode_hamiltonian_ED@rand_vector
        ed_mat_squared_vec = single_mode_hamiltonian_ED@single_mode_hamiltonian_ED@rand_vector
        
        ci_mat_vec = single_mode_hamiltonian_construction_residual_vec(model,file_name,order,nZ,rand_vector)
        ci_mat_squared_vec = single_mode_hamiltonian_construction_squared_residual_vec(model, file_name,
                                                                                    order, nZ,
                                                                                    rand_vector)
        
    elif hamiltonian.N == 2:
        double_mode_hamiltonian_ED = exact_FCI_2_mode_hamiltonian(model, file_name, order, nZ)
        rand_vector = np.random.randint(1,100,size=double_mode_hamiltonian_ED.shape[0])
        
        ed_mat_vec = double_mode_hamiltonian_ED@rand_vector
        ed_mat_squared_vec = double_mode_hamiltonian_ED@double_mode_hamiltonian_ED@rand_vector
        
        ci_mat_vec = double_mode_hamiltonian_residual_vec(model,file_name,order,nZ,rand_vector)
        ci_mat_squared_vec = double_mode_hamiltonian_residual_squared_vec(model,file_name,order,nZ,rand_vector)
    else:
        assert  hamiltonian.N != 1 and hamiltonian.N != 2, 'The residual can only be determined to be correct for a single or 2 mode system'  
    
    # breakpoint()
    lines = ['Results hamiltonian-vector multiplication',
             f'number of modes: {hamiltonian.N}',
             f'hamiltonian rank: {order}',
             f'vector: {rand_vector}',
             f'Hamiltonian matrix vector multiplication via ED: {ed_mat_vec.real}',
             f'Hamiltonian matrix vector multiplication via generated Z eqautions: {ci_mat_vec.real}',
             f'Difference to see if Hamiltonian matrix vector multiplication are same between methods: {np.abs(ed_mat_vec-ci_mat_vec.real)}',
             f'Hamiltonian squared matrix vector multiplication via ED: {ed_mat_squared_vec.real}',
             f'Hamiltonian matrix squared vector multiplication via generated Z eqautions: {ci_mat_squared_vec.real}',  
             f'Difference to see if Hamiltonian matrix squared vector  multiplication are same between methods: {np.abs(ed_mat_squared_vec-ci_mat_squared_vec.real)}',
             ]
    text = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)     
    
    return()
 
def generate_cc_acf_data_oz(model, file_name, order, nZ,t_final=10.0, ext = None,dist = 0,specd=[],output_path=None,q=False,z_chk=False,norm=False):
    """ x """

    # initialize object 
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=nZ,
        calculate_population_flag=False,init_disp=float(dist),disps=specd,
        correlation_funcs = q, z_check = z_chk,include_norm=norm
    )
    
    # Call integration on object 
    start_time = time.time()
    
    nof_steps=int(1e4)
    hamiltonian.rk45_integration_oz(t_final=t_final, nof_points=nof_steps)
    
    end_time = time.time()
    print(f"Very simple timing:   {end_time - start_time:10.4f}")

    if ext != None:
        file_name = file_name+f'_h{order}_Z{nZ}_T{t_final}_{ext}'    
    else:
        file_name = file_name+f'_h{order}_Z{nZ}_T{t_final}'
    
    # make plot and store data
    non_normalized_path = hamiltonian.save_acf_data(file_name=file_name, output_path=output_path)
    # breakpoint()
    print('output_path_ABS',output_path)
    

    if hamiltonian.correlation_funcs:
    
        file_path_H_ABS = join(output_path,f"exp_H_{file_name}.txt")
        hamiltonian._save_data(file_path_H_ABS, hamiltonian.t_cc, hamiltonian.exp_H_tau_cc_ABS)

        file_path_norm_ABS = join(output_path,f"norm_{file_name}.txt")
        hamiltonian._save_data(file_path_norm_ABS, hamiltonian.t_cc, hamiltonian.norm_tau_cc_ABS)

        # interpolating the corr funcs will be important

        print("-"*40 + "\nInterpolating H ABS\n" + "-"*40 + "\n")
        normalized_path_ABS = project.spectra.generate_normalized_acf_results(
                dirname(file_path_H_ABS),
                basename(file_path_H_ABS),
                None,
                output_path_norm_cc=output_path,
                mctdh_t_final=t_final*0.5,
                mctdh_dt=0.1
            )
        
        print("-"*40 + "\nInterpolating norm ABS\n" + "-"*40 + "\n")
        normalized_path_ABS = project.spectra.generate_normalized_acf_results(
                dirname(file_path_norm_ABS),
                basename(file_path_norm_ABS),
                None,
                output_path_norm_cc=output_path,
                mctdh_t_final=t_final*0.5,
                mctdh_dt=0.1
            )
        
        if hamiltonian.N == 1:
            file_path_q_ABS = join(output_path,f"exp_q_{file_name}.txt")
            hamiltonian._save_data(file_path_q_ABS, hamiltonian.t_cc, hamiltonian.q_tau_cc_ABS)
            
            file_path_q2_ABS = join(output_path,f"exp_q_2_{file_name}.txt")
            hamiltonian._save_data(file_path_q2_ABS, hamiltonian.t_cc, hamiltonian.q_2_tau_cc_ABS)

            # interpolating the corr funcs will be important
            # interpolate for ACF(ABS)
            print("-"*40 + "\nInterpolating q ABS\n" + "-"*40 + "\n")
            normalized_path_ABS = project.spectra.generate_normalized_acf_results(
                    dirname(file_path_q_ABS),
                    basename(file_path_q_ABS),
                    None,
                    output_path_norm_cc=output_path,
                    mctdh_t_final=t_final*0.5,
                    mctdh_dt=0.1
                )
            
            print("-"*40 + "\nInterpolating q2 ABS\n" + "-"*40 + "\n")
            normalized_path_ABS = project.spectra.generate_normalized_acf_results(
                    dirname(file_path_q2_ABS),
                    basename(file_path_q2_ABS),
                    None,
                    output_path_norm_cc=output_path,
                    mctdh_t_final=t_final*0.5,
                    mctdh_dt=0.1
                )
        

    # interpolate for ACF(ABS)
    print("-"*40 + "\nInterpolating ABS\n" + "-"*40 + "\n")
    normalized_path_ABS = project.spectra.generate_normalized_acf_results(
            dirname(non_normalized_path),
            basename(non_normalized_path),
            None,
            output_path_norm_cc=output_path,
            mctdh_t_final=t_final*0.5,
            mctdh_dt=0.1
        )

    # Create T_1 time data file
    if model[VMK.N] == 1:
        file_path_ABS = join(output_path,f"T1_{file_name}.txt")
        hamiltonian._save_data(file_path_ABS, hamiltonian.t_cc, hamiltonian.T1_tau_cc_ABS)
    
    file_path_T1 = join(output_path,f"T1_{file_name}.csv")
    T1_times = np.asarray(hamiltonian.t_cc).reshape(-1, 1)
    T1_data = np.hstack([T1_times,hamiltonian.T1_tau_cc])
    np.savetxt(file_path_T1,T1_data,delimiter=",",
    header="time," + ",".join(f"T{i}" for i in range(hamiltonian.N)),
    comments="")
      
    return(normalized_path_ABS) 


def generate_ed_results(model,file_name,order,HO_basis_size,t_final=10.0,subspace_calc=False,q_eval=False,ext=None,output_path=None):
    
    hamiltonian = vibronic_hamiltonian(
        model, file_name,
        hamiltonian_truncation_order=order, cc_truncation_order=order,
        T_truncation_order=1, Z_truncation_order=1,
        calculate_population_flag=False,normal_order=True
    )
    # pathing for outputs 
    
    assert hamiltonian.N == 1 or hamiltonian.N == 2, 'only valid for 1d and 2d systems'
    
    # do ED routine for getting Q online in single mode case
    if q_eval == True and hamiltonian.N == 1:
        eigvals,time,acf,Q_exp_val,Q2_exp_val,H_mat_HO_basis,intensities,eig_q_results, exact_animation, H_exp_val, norm_val = hamiltonian.ED_subroutine(HO_basis_size,
                                                                                          t_final,
                                                                                          stepsize=0.1,
                                                                                          subspace_calc=subspace_calc,
                                                                                          q_eval=q_eval)
        q_file_name = 'q_'+file_name   
        q_output_path = hamiltonian.save_acf_data(file_name=q_file_name, 
                                                output_path=output_path,
                                                time=time,
                                                acf_ABS=Q_exp_val)
        q2_file_name = 'q2_' + file_name
        q2_output_path = hamiltonian.save_acf_data(file_name=q2_file_name, 
                                                output_path=output_path,
                                                time=time,
                                                acf_ABS=Q2_exp_val)
        exp_H_file_name = 'exp_H_'+file_name   
        exp_H_output_path = hamiltonian.save_acf_data(file_name=exp_H_file_name, 
                                                output_path=output_path,
                                                time=time,
                                                acf_ABS=H_exp_val)
        norm_file_name = 'norm_' + file_name
        norm_output_path = hamiltonian.save_acf_data(file_name=norm_file_name, 
                                                output_path=output_path,
                                                time=time,
                                                acf_ABS=norm_val)
        
        eig_q_file_name = 'qE_eig' + file_name  
        eig_q_results.to_csv(join(output_path,f'eig_q_{file_name}'),index=False)
        animation_file_name = f'anime_{file_name}.gif'
        exact_animation.save(join(output_path,animation_file_name),writer='ffmpeg')

        
    else:
        eigvals,time,acf,H_mat_HO_basis,intensities = hamiltonian.ED_subroutine(HO_basis_size,
                                                                            t_final,
                                                                            stepsize=0.1,
                                                                            subspace_calc=subspace_calc)
    eigenvalue_file_name = 'eig'+file_name     
    hamiltonian._save_real_data(join(output_path,eigenvalue_file_name),eigvals,
                                np.real(intensities),
                                'Eigenvalue',
                                'Associated Ground Survival Intensity') 
    
    acf_output_path = hamiltonian.save_acf_data(file_name=file_name,
                              output_path=output_path,
                              time=time,
                              acf_ABS=acf)
        
    return(H_mat_HO_basis)


