#!/bin/bash
cd "${0%/*}"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate trace_analysis
git pull --ff-only
conda deactivate
conda update conda
conda env update --file environment.yml --prune
