#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=../Results/ED_results/double_well_tests/results
mkdir -p ${LOGDIR}

echo "Logging to ${LOGDIR}"
echo "Starting at $(date)"

python3 testing.py CCEEED -f H47b -o 4 -t 200 -HO 400 -q -dir ${LOGDIR} 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47b_HO400_T200.log

python3 testing.py CCEEED -f H47c -o 4 -t 200 -HO 400 -q -dir ${LOGDIR} 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47c_HO400_T200.log

python3 testing.py CCEEED -f H47d -o 4 -t 200 -HO 400 -q -dir ${LOGDIR} 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47d_HO400_T200.log

python3 testing.py CCEEED -f H47e -o 4 -t 200 -HO 400 -q -dir ${LOGDIR} 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47e_HO400_T200.log

python3 testing.py CCEEED -f H47f -o 4 -t 200 -HO 400 -q -dir ${LOGDIR} 2>&1 | tee ${LOGDIR}/${TIMESTAMP}H47f_HO400_T200.log
