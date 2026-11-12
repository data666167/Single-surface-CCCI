#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/CC_results/double_well_tests/results
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"


python3 testing.py CCP -f H47b -o 4 -z 4 -t 200 -q -dir ${LOGDIR} -d 0   2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47b_z4_T200_d0.log
python3 testing.py CCP -f H47b -o 4 -z 4 -t 200 -q -dir ${LOGDIR} -d 7   2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47b_z4_T200_d7.log
python3 testing.py CCP -f H47b -o 4 -z 4 -t 200 -q -dir ${LOGDIR} -d -7  2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47b_z4_T200_d-7.log
python3 testing.py CCP -f H47b -o 4 -z 4 -t 200 -q -dir ${LOGDIR} -d -14 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47b_z4_T200_d-14.log


