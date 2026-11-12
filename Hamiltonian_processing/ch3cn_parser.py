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

df = pd.read_csv(
    "ch3cn.csv",
    sep=r"\s+",      # one or more spaces
    header=None,     # no header row in your sample
    names=["label", "value_cm"]
)

parts = df["label"].str.split("_", n=4, expand=True)
df[["prefix", "i", "j", "k", "l"]] = parts[[0, 1, 2, 3, 4]]
df[["i", "j", "k", "l"]] = df[["i", "j", "k", "l"]].fillna(0)
value_ev =1/8065.61042 # 8065.61042 cm^{-1}
df["value_ev"] = df["value_cm"]*value_ev

# diagonal elements and their indexes
mask_cubic = df['prefix'] == 'CB'  
for idx, row in df[mask_cubic].iterrows():
    df.at[idx, "j"] = int(row['i'])
    df.at[idx, "k"] = int(row['i'])

mask_quartic  = df['prefix'] == 'QR'  
for idx, row in df[mask_quartic].iterrows():
    df.at[idx, "j"] = int(row['i'])
    df.at[idx, "k"] = int(row['i'])
    df.at[idx, "l"] = int(row['i'])

df.to_csv("ch3cn_parsed.csv", index=False)

# Construct the model parameters for json and MCTDH

# initializing the coefficient arrays
number_of_modes = 12
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

# adding variables to shit
maskw = df['prefix'] == 'w'  
mask3 = (df[['i','j','k','l']] != 0).sum(axis=1) == 3
mask4 = (df[['i','j','k','l']] != 0).sum(axis=1) == 4

for _, row in df[maskw].iterrows():
    # breakpoint()
    idx = (int(int(row['i']) - 1))
    test_dict[VMK.w][idx] = row['value_ev']
    
for _, row in df[mask3].iterrows():
    idx = (int(int(row['i']) - 1),int(int(row['j'])) - 1,int(int(row['k']) - 1))    
    cubic_coeffs[idx[0], idx[1], idx[2], :, :] = row['value_ev']

for _, row in df[mask4].iterrows():
    idx = (int(int(row['i']) - 1),int(int(row['j']) - 1),int(int(row['k']) - 1),int(int(row['l']) - 1))    
    quartic_coeffs[idx[0], idx[1], idx[2], idx[3],:, :] = row['value_ev']

file_name =  'ch3cn_ev'

test_dict[VMK.G1] = linear_coeffs
test_dict[VMK.G2] = quadratic_coeffs
test_dict[VMK.G3] = cubic_coeffs
test_dict[VMK.G4] = quartic_coeffs

file_path_json = join(os.getcwd(),file_name+'.json')
file_path_op = join(os.getcwd(),file_name+'.op')

vIO.save_model_to_JSON(file_path_json,test_dict)
vIO.write_raw_model_op_file(file_path_op,test_dict,4)

print()





breakpoint()