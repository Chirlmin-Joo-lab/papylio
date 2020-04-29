ECHO OFF
CALL conda activate trace_analysis
git pull --ff-only
CALL conda deactivate
CALL conda update conda
CALL conda env update --file environment.yml --prune
pause
