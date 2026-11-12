#!/bin/bash

# many mode analysis
# python3 -m tabulations.many_mode_1000fs_extraction
# python3 -m tabulations.many_mode_1000fs_compilation

# single mode analysis
python3 -m tabulations.single_mode_extraction
python3 -m tabulations.single_mode_compilation