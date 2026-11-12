#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/CH2NH_single_mode/results
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d5M1 -dir ${LOGDIR} -sd 0.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d5M1.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d5M2 -dir ${LOGDIR} -sd 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d5M2.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d5M3 -dir ${LOGDIR} -sd 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d5M3.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M4 -dir ${LOGDIR} -sd 0 0 0 1 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M4.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M5 -dir ${LOGDIR} -sd 0 0 0 0 1 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M5.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M6 -dir ${LOGDIR} -sd 0 0 0 0 0 1 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M6.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M7 -dir ${LOGDIR} -sd 0 0 0 0 0 0 1 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M7.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M8 -dir ${LOGDIR} -sd 0 0 0 0 0 0 0 1 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M8.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 4 -t 500 -e d1M9 -dir ${LOGDIR} -sd 0 0 0 0 0 0 0 0 1 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z4_t500_d1M9.log


python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d5M1 -dir ${LOGDIR} -sd 0.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d5M1.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d5M2 -dir ${LOGDIR} -sd 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d5M2.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d5M3 -dir ${LOGDIR} -sd 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d5M3.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M4 -dir ${LOGDIR} -sd 0 0 0 1 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M4.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M5 -dir ${LOGDIR} -sd 0 0 0 0 1 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M5.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M6 -dir ${LOGDIR} -sd 0 0 0 0 0 1 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M6.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M7 -dir ${LOGDIR} -sd 0 0 0 0 0 0 1 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M7.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M8 -dir ${LOGDIR} -sd 0 0 0 0 0 0 0 1 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M8.log
python3 testing.py CCP -f ch2nh_evp -o 4 -z 2 -t 500 -e d1M9 -dir ${LOGDIR} -sd 0 0 0 0 0 0 0 0 1 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch2nh_evp_z2_t500_d1M9.log