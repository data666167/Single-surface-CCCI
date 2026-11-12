#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=/home/oz_computron/Desktop/Research_idea/single_surface_vecc/Results/CC_results/reproduce_single_mode_${TIMESTAMP}
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d5M1  -sd 0.5 0 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d5M1.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M2  -sd 0 1.5 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M2.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M3  -sd 0 0 1.5 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M3.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M4  -sd 0 0 0 1.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M4.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d5M5  -sd 0 0 0 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d5M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d5M6  -sd 0 0 0 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d5M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M7  -sd 0 0 0 0 0 0 1.5 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M7.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M8  -sd 0 0 0 0 0 0 0 1.5 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M8.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M9  -sd 0 0 0 0 0 0 0 0 1.5 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M9.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M10 -sd 0 0 0 0 0 0 0 0 0 1.5 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M10.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M11 -sd 0 0 0 0 0 0 0 0 0 0 1.5 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M11.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 2 -t 500 -e d15M12 -sd 0 0 0 0 0 0 0 0 0 0 0 1.5 2>&1 | tee ${LOGDIR}/ch3cn_ev_z2_t500_d15M12.log

python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d5M1  -sd 0.5 0 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d5M1.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M2  -sd 0 1.5 0 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M2.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M3  -sd 0 0 1.5 0 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M3.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M4  -sd 0 0 0 1.5 0 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M4.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d5M5  -sd 0 0 0 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d5M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d5M6  -sd 0 0 0 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d5M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M7  -sd 0 0 0 0 0 0 1.5 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M7.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M8  -sd 0 0 0 0 0 0 0 1.5 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M8.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M9  -sd 0 0 0 0 0 0 0 0 1.5 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M9.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M10 -sd 0 0 0 0 0 0 0 0 0 1.5 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M10.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M11 -sd 0 0 0 0 0 0 0 0 0 0 1.5 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M11.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d15M12 -sd 0 0 0 0 0 0 0 0 0 0 0 1.5 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d15M12.log


