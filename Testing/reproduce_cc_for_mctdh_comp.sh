#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/reproduce_cc_for_mctdh
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"


python3 testing.py CCP -f ch3cn_ev  -o 4 -z 4 -t 200 -d 0      -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t200_d0.log 
python3 testing.py CCP -f ch3cn_ev  -o 4 -z 4 -t 200 -d 0.5    -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t200_d5.log
python3 testing.py CCP -f ch3cn_ev  -o 4 -z 4 -t 200 -d 0.7071 -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t200_d7071.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 200 -d 0      -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch2nh_ev_z4_t200_d0.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 200 -d 0.5    -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch2nh_ev_z4_t200_d5.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 200 -d 0.7071 -dir ${LOGDIR} 2>&1 |tee ${LOGDIR}/${TIMESTAMP}ch2nh_ev_z4_t200_d7071.log

