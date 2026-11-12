#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/CH3CN_single_mode/results
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d5M1  -sd 0.5 0 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d5M1.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M2  -sd 0 1.5 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M2.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M3  -sd 0 0 1.5 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M3.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M4  -sd 0 0 0 1.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M4.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d5M5  -sd 0 0 0 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d5M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d5M6  -sd 0 0 0 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d5M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M7  -sd 0 0 0 0 0 0 1.5 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M7.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M8  -sd 0 0 0 0 0 0 0 1.5 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M8.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M9  -sd 0 0 0 0 0 0 0 0 1.5 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M9.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M10 -sd 0 0 0 0 0 0 0 0 0 1.5 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M10.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M11 -sd 0 0 0 0 0 0 0 0 0 0 1.5 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M11.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -dir ${LOGDIR} -e d15M12 -sd 0 0 0 0 0 0 0 0 0 0 0 1.5 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z2_t500_d15M12.log

python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d5M1  -sd 0.5 0 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d5M1.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M2  -sd 0 1.5 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M2.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M3  -sd 0 0 1.5 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M3.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M4  -sd 0 0 0 1.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M4.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d5M5  -sd 0 0 0 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d5M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d5M6  -sd 0 0 0 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d5M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M7  -sd 0 0 0 0 0 0 1.5 0 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M7.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M8  -sd 0 0 0 0 0 0 0 1.5 0 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M8.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M9  -sd 0 0 0 0 0 0 0 0 1.5 0 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M9.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M10 -sd 0 0 0 0 0 0 0 0 0 1.5 0 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M10.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M11 -sd 0 0 0 0 0 0 0 0 0 0 1.5 0 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M11.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -dir ${LOGDIR} -e d15M12 -sd 0 0 0 0 0 0 0 0 0 0 0 1.5 2>&1 | tee ${LOGDIR}/${TIMESTAMP}ch3cn_ev_z4_t500_d15M12.log


