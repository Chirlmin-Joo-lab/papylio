#!/bin/bash
cd "${0%/*}"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate trace_analysis
pythonw GUI.py
conda deactivate

#python tt.py
