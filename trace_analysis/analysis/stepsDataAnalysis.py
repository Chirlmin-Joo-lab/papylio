# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:05:13 2019

@author: iason
"""

import concurrent.futures
import time
import numpy as np
import pandas as pd
import os

def find_mol_dwells(mol, trace='red'):
    max_time = mol.file.time[-1]
    exp_time = mol.file.exposure_time

    times = mol.steps.time.values[mol.steps.trace == trace]
    try:
        times1 = times.reshape((int(times.size/2), 2))
    except ValueError:
        return

    offtimes = np.diff(times1, axis=1).flatten()

    labels = []
    for i, d in enumerate(offtimes):
        lab = 'm'
        if times[0] == 0 and i == 0:  # first loop
            lab = 'l'
        if max_time - times[-1] < 0.1 and i == len(offtimes) - 1:  # last loop
            lab = 'r'
        labels.append(lab)
    offtimes = pd.DataFrame({'offtime': offtimes, 'side': labels})

    # Calculate the on times
    ontimes = []
    labels = []
    if times[0] > exp_time:  # append the left kon if it exists
        ontimes.append(times[0])
        labels.append('l')

    for i in range(2, times.size, 2):
        ontimes.append(times[i] - times[i-1])
        labels.append('m')

    if max_time - times[-1] > exp_time:  # append the right kon if it exists
        ontimes.append(max_time - times[-1])
        labels.append('r')

    ontimes = pd.DataFrame({'ontime': ontimes,
                            'onside': labels,
                            'order': np.arange(0, len(ontimes))})

  # Calculate the average FRET for each dwell

    avg_fret = []
    Icheck = int((mol.steps.Imin.tail(1)== mol.steps.Imin[0]) &
                  (mol.steps.Iroff.tail(1) == mol.steps.Iroff[0]) &
                  (mol.steps.Igoff.tail(1) == mol.steps.Igoff[0]))
    if Icheck == 1:  #  check if thresholds the same for each dwell of the molecule
        fret = mol.E(Imin=mol.steps.Imin[0], Iroff=mol.steps.Iroff[0], Igoff=mol.steps.Igoff[0])
    else:
        print(f'Ioffsets are not equal for molecule:{i+1}')
        fret = []

    for ii in range(0, len(times)):
        if ii % 2 != 0:
            istart = int(round(times[ii-1]/exp_time))
            iend = int(round(times[ii]/exp_time))
            a_fret = round(np.mean(fret[istart:iend]), 2)
            if (a_fret <= 1 and a_fret >= 0):
                avg_fret.append(a_fret)
            else:
                avg_fret.append(0)
                print(f'FRET corrupted for molecule:{i+1}')
        # else:
        #     avg_fret.append(None)

    avgFRET = pd.DataFrame({'avrgFRET': avg_fret})

    return pd.concat([offtimes, ontimes, avgFRET], axis=1)


def process_molecule(mol):
    traces_unique = pd.unique(mol.steps.trace.values)
    results = []
    for trace in traces_unique:

        results.append(find_mol_dwells(mol, trace=trace))

    result = pd.concat(results, axis=0, ignore_index=True)
    mol.steps = pd.concat([mol.steps, result], axis=1)
    return mol.steps

def analyze_steps(file, save=True):
    molecules_with_data = [mol for mol in file.molecules if mol.steps is not None]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(process_molecule, mol)
                   for mol in molecules_with_data]
    # if the results need to be manipulated uncomments the following lines:
    # for f in concurrent.futures.as_completed(results):
    #     print(f.result())

    filename = file.name+'_dwells_data.xlsx'
    data = file.savetoExcel(filename=filename, save=save)

if __name__ == '__main__':

    import sys
    from pathlib import Path
    p = Path(__file__).parents[2]
    sys.path.insert(0, str(p))

    from trace_analysis import Experiment

    start = time.time()
    mainPath='F:/Google Drive/PhD/Programming - Data Analysis/traceanalysis/traces'
    exp = Experiment(mainPath)
    file = exp.files[0]

    data = analyze_steps(file)


    print(f'Processed step data in {time.time() - start:.2f} sec')


