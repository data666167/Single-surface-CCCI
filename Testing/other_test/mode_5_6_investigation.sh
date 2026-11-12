#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=/home/oz_computron/Desktop/Research_idea/single_surface_vecc/Results/CC_results/mode_5_6_investigation_${TIMESTAMP}
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

# investigation
# mode 9 is included as a control to compare against mode 6 and 5    
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d1M5  -sd 0 0 0 0 1 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d1M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d1M6  -sd 0 0 0 0 0 1 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d1M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d1M9  -sd 0 0 0 0 0 0 0 0 1 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d1M9.log

python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.5M5  -sd 0 0 0 0 0.5 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.5M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.5M6  -sd 0 0 0 0 0 0.5 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.5M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.5M9  -sd 0 0 0 0 0 0 0 0 0.5 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.5M9.log

python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.2M5  -sd 0 0 0 0 0.2 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.2M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.2M6  -sd 0 0 0 0 0 0.2 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.2M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.2M9  -sd 0 0 0 0 0 0 0 0 0.2 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.2M9.log

python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.1M5  -sd 0 0 0 0 0.1 0 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.1M5.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.1M6  -sd 0 0 0 0 0 0.1 0 0 0 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.1M6.log
python3 testing.py CCP -f ch3cn_ev -o 4 -z 4 -t 500 -e d.1M9  -sd 0 0 0 0 0 0 0 0 0.1 0 0 0 2>&1 | tee ${LOGDIR}/ch3cn_ev_z4_t500_d.1M9.log
