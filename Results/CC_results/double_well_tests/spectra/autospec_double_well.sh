#!/bin/bash
set -e

spec_dir="$PWD"

(
    cd ../results/model_H47b_rank_4_Z_4_time_200_dist_-7
    autospec86 -o spec.pl -f ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
    mv spec.pl "$spec_dir/case_1_90_CC_H47b.pl"
)

(
    cd ../results/model_H47b_rank_4_Z_4_time_200_dist_7
    autospec86 -o spec.pl -f ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
    mv spec.pl "$spec_dir/case_2_90_CC_H47b.pl"
)

(
    cd ../results/model_H47b_rank_4_Z_4_time_200_dist_-14
    autospec86 -o spec.pl -f ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 10 2
    mv spec.pl "$spec_dir/case_3_10_CC_H47b.pl"
)

(
    cd ../results/model_H47b_rank_4_Z_4_time_200_dist_0
    autospec86 -o spec.pl -f ACF_ABS_CC_H47b_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 10 2
    mv spec.pl "$spec_dir/case_4_10_CC_H47b.pl"
)