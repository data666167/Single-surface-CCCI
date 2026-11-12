#!/bin/bash

autospec86 -o ./case_1_90_ED_H47d.pl -f ../results/model_H47d_rank_4_basis_400_time_200/ACF_ABS_CC_H47d.txt -p 8000 10 -5 ev 90 2
autospec86 -o ./case_2_90_ED_H47e.pl -f ../results/model_H47e_rank_4_basis_400_time_200/ACF_ABS_CC_H47e.txt -p 8000 10 -5 ev 90 2
autospec86 -o ./case_3_10_ED_H47f.pl -f ../results/model_H47f_rank_4_basis_400_time_200/ACF_ABS_CC_H47f.txt -p 8000 10 -5 ev 10 2
autospec86 -o ./case_4_10_ED_H47b.pl -f ../results/model_H47b_rank_4_basis_400_time_200/ACF_ABS_CC_H47b.txt -p 8000 10 -5 ev 10 2
