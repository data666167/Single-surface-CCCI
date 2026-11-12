#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/CH2NH_trial_1000/results
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

python3 testing.py CCP -f ch2nh_evp  -o 4 -z 4 -t 1000 -d 0.5 -dir ${LOGDIR}  -q  2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch2nh_ev_z4_t1000_d5.log