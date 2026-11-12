#!/bin/bash

# double well results
bash ED_double_well.sh
bash CC_double_well.sh

# molecular results
bash reproduce_single_mode_CH3CN.sh
bash trial_1000_CH3CN.sh

bash reproduce_single_mode_CH2NH.sh
bash trial_1000_CH2NH.sh


# 200 fs propagation for 3 displacements for both chemical systems 
bash reproduce_cc_for_mctdh_comp.sh