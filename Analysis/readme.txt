You must run the analysis scripts in the analysis directory using bash: 

ex. bash molecular_analysis.sh 

All other python scripts should from ./plotting,./signal_processing and ./tabulations must 
be run in the analsysis directory:

python3 -m plotting.double_well_plotter

The analysis scripts should be run after propagation if results must be recomputed 
please copy the current relative directory structure saving the previous calculations 