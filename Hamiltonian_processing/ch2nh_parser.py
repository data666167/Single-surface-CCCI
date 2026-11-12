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
import random as rn
import pandas as pd
from math import factorial

# third party imports
import numpy as np
import matplotlib.pyplot as plt


# import the path to the package
project_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import project

# local imports
from project.vibronic_hamiltonian import vibronic_hamiltonian
from project.vibronic import vIO, VMK, model_op
import project.spectra
import project.log_conf 
df = pd.read_csv("ch2nh_pot.dat", 
                 sep=r'\s+',
                 header=None)

# 9 exponents + 1 coefficient
df.columns = [f"n{i}" for i in range(1, 10)] + ["coeff"]
value_ev =1/8065.61042
df['coeffs_ev'] = df["coeff"]*value_ev
df["total_order"] = df[[f"n{i}" for i in range(1, 10)]].sum(axis=1)

# assigning ijkl 
# iterate through the rows 
# for each row check each column from left to right 
    # for each column check the number 
        # if number is non zero  
            # check number call it multiplicty 
                # append the index of the associated column 
                # to the number of columns of the ijkl columns 
                #  for the row starting from the i,j,k,l
                # i.e. col 1 has multiplicty of 4 the i = 1, j = 1, k = 1 , l = 1
                # if there multiple numbers in a given row do not overwrite i,j,k,l
                # append the column indez to the multiplicty of nonzero ijkl 
                # i.e  col 5 had 2 col 7 has 1 then the ijkl for that row should be i = 5, j = 5, k = 7 , l = 0  
        ## safeties 
        # check that there are no more than 4 non-zero numbers in a given row from the column 1-10
        # check that sum of the numbers of the columns 0-10 is not greater than 4  

def assign_ijkl(row):
    # Collect mode indices with multiplicity, left to right
    idxs = []
    for i in range(1, 10):          # columns n1..n9
        n = int(row[f"n{i}"])
        if n > 0:
            idxs.extend([i] * n)

    # Safety: total power <= 4
    total_power = sum(int(row[f"n{i}"]) for i in range(1, 10))
    if total_power > 4:
        raise ValueError(f"Total power > 4 in row:\n{row}")

    # Safety: no more than 4 nonzero slots after expansion
    if len(idxs) > 4:
        raise ValueError(f"More than 4 ijkl indices in row:\n{row}")

    # Pad with zeros to length 4
    while len(idxs) < 4:
        idxs.append(0)

    i, j, k, l = idxs[0], idxs[1], idxs[2], idxs[3]
    return pd.Series({"i": i, "j": j, "k": k, "l": l})

def symmetry_multiplicity(row):
    # total power n
    n = 0
    denom = 1
    for k in range(1, 10):
        nk = int(row[f"n{k}"])
        n += nk
        denom *= factorial(nk)
    if n == 0:
        return 1
    return factorial(n) // denom

# Apply to all rows to create i,j,k,l columns
ijkl = df.apply(assign_ijkl, axis=1)
df = pd.concat([df, ijkl], axis=1)

# find permutations using multiplicty 
df["multiplicity"] = df.apply(symmetry_multiplicity, axis=1)

df.to_csv("ch2nh_parsed.csv", index=False)

# Construct the model parameters for json and MCTDH

# initializing the coefficient arrays
number_of_modes = 9
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

linear_coeffs = np.zeros((number_of_modes, surfaces, surfaces))
quadratic_coeffs = np.zeros((number_of_modes, number_of_modes, surfaces, surfaces))
cubic_coeffs = np.zeros((number_of_modes, number_of_modes, number_of_modes, surfaces, surfaces))
quartic_coeffs = np.zeros((number_of_modes, number_of_modes, number_of_modes, number_of_modes, surfaces, surfaces))
test_dict[VMK.w] = np.zeros([number_of_modes])

maskw = df['total_order'] == 2  
mask3 = df['total_order'] == 3
mask4 = df['total_order'] == 4

for _, row in df[maskw].iterrows():
    # breakpoint()
    idx = (int(int(row['i']) - 1))
    test_dict[VMK.w][idx] = row['coeffs_ev']
    
for _, row in df[mask3].iterrows():
    idx = (int(int(row['i']) - 1),int(int(row['j'])) - 1,int(int(row['k']) - 1))    
    cubic_coeffs[idx[0], idx[1], idx[2], :, :] = row['coeffs_ev']*(row['multiplicity']/6)

for _, row in df[mask4].iterrows():
    idx = (int(int(row['i']) - 1),int(int(row['j']) - 1),int(int(row['k']) - 1),int(int(row['l']) - 1))    
    quartic_coeffs[idx[0], idx[1], idx[2], idx[3],:, :] = row['coeffs_ev']*(row['multiplicity']/24)

file_name =  'ch2nh_ev_perm'

test_dict[VMK.G1] = linear_coeffs
test_dict[VMK.G2] = quadratic_coeffs
test_dict[VMK.G3] = cubic_coeffs
test_dict[VMK.G4] = quartic_coeffs

file_path_json = join(os.getcwd(),file_name+'.json')
file_path_op = join(os.getcwd(),file_name+'.op')

vIO.save_model_to_JSON(file_path_json,test_dict)
vIO.write_raw_model_op_file(file_path_op,test_dict,4)


breakpoint()