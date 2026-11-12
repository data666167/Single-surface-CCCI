To run all calculations as seen in the current directory structure run the master script
using:

bash master_script.sh

The master_script and all test scripts, and tests must be conducted from the testing directory.
To repopulate the results present in the directories, please copy the same results directory structure

To run new propagations for novel systems refer to the syntax for CC/CI propagation 
in reproduce_cc_for_mctdh_comp.sh for a displacement of all modes and for mode specific propagations
consider the syntax in reproduce_single_mode_CH3CN.sh.

Analysis scripts that populate tabulated_results subdirectories in results only populate 
after running the analysis directory specific scripts. Please read the corresponding readme

For new hamiltonians the syntax of the corresponding json for cc/ci propagation can be found 
in the CC hamiltonians_processing.
