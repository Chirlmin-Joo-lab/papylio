#!/bin/bash
cd "${0%/*}"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate papylio
python -m papylio
conda deactivate

#python tt.py
  