#!/bin/bash
set -e

spec_dir="$PWD"

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0/
autospec86 -o ./ch2nh_40_d0.pl  -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch2nh_40_d0.pl "$spec_dir/ch2nh_40_d0.pl"
)

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0.5/
autospec86 -o ./ch2nh_40_d.5.pl  -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch2nh_40_d.5.pl "$spec_dir/ch2nh_40_d.5.pl"
)

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0.7071/
autospec86 -o ./ch2nh_40_d.7071.pl  -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch2nh_40_d.7071.pl "$spec_dir/ch2nh_40_d.7071.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0/
autospec86 -o ./ch3cn_40_d0.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch3cn_40_d0.pl "$spec_dir/ch3cn_40_d0.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0.5/
autospec86 -o ./ch3cn_40_d.5.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch3cn_40_d.5.pl "$spec_dir/ch3cn_40_d.5.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0.7071/
autospec86 -o ./ch3cn_40_d.7071.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 40 2
mv ch3cn_40_d.7071.pl "$spec_dir/ch3cn_40_d.7071.pl"
)

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0/
autospec86 -o  ./ch2nh_90_d0.pl  -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch2nh_90_d0.pl "$spec_dir/ch2nh_90_d0.pl"
)

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0.5/
autospec86 -o  ./ch2nh_90_d.5.pl -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch2nh_90_d.5.pl "$spec_dir/ch2nh_90_d.5.pl"
)

(
cd ../model_ch2nh_evp_rank_4_Z_4_time_200_dist_0.7071/
autospec86 -o  ./ch2nh_90_d.7071.pl  -f ACF_ABS_CC_ch2nh_evp_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch2nh_90_d.7071.pl "$spec_dir/ch2nh_90_d.7071.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0/
autospec86 -o  ./ch3cn_90_d0.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch3cn_90_d0.pl "$spec_dir/ch3cn_90_d0.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0.5/
autospec86 -o  ./ch3cn_90_d.5.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch3cn_90_d.5.pl "$spec_dir/ch3cn_90_d.5.pl"
)

(
cd ../model_ch3cn_ev_rank_4_Z_4_time_200_dist_0.7071/
autospec86 -o  ./ch3cn_90_d.7071.pl  -f ACF_ABS_CC_ch3cn_ev_h4_Z4_T200.0_normalized.txt -p 8000 10 -5 ev 90 2
mv ch3cn_90_d.7071.pl "$spec_dir/ch3cn_90_d.7071.pl"
)