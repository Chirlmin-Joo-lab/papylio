#!/bin/bash
cd "${0%/*}"
if [ -d .git ]
then
	echo "Already a git repository present"
else
	git clone https://gitlab.tudelft.nl/ivoseverins/traceanalysis.git
	CONDA_BASE=$(conda info --base)
	source $CONDA_BASE/etc/profile.d/conda.sh
	conda env create -f traceAnalysis/environment.yml
fi