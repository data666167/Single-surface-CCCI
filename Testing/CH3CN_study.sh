#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/CH3CN_study
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

# Z_1 stays zero only T_1 propagates
python3 testing.py CCP -f ch3cn_ev  -o 4 -z 1 -t 200 -d 0.7071 -dir ${LOGDIR}  -q  2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z1_t200_d.7071.log
# Z_chk sets the T_1 derivative to 0 so T_1 is not propagated and only CI changes in time
python3 testing.py CCP -f ch3cn_ev  -o 4 -z 4 -t 200 -d 0.7071  -zchk -dir ${LOGDIR} -e no_T_1  -q  2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t200_d.7071_no_T_1.log
# Z_2 comparison 
python3 testing.py CCP -f ch3cn_ev  -o 4 -z 2 -t 200 -d 0.7071  -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t200_d.7071.log