# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:05:13 2019

@author: iason
"""

import concurrent.futures
import time
import numpy as np
import pandas as pd


def find_mol_dwells(mol, trace='red'):
    max_time = mol.file.time[-1]
    exp_time = mol.file.exposure_time

    times = mol.steps.time.values[mol.steps.trace == trace]
    
    try:
        times1 = times.reshape((int(times.size/2), 2))
    except ValueError:
        print(f'Uneven number of clicks for mole {mol}')
        return

    # Calculate the offtimes and assign labels
    offtimes = np.NaN*np.ones(times.size)
    offlabels = np.empty(times.size, dtype=str)
    for i in range(0, times.size, 2):
        offtimes[i] = times[i+1] - times[i]
        lab = 'm'
        if times[0] < 1 and i == 0:  # first loop
            lab = 'l'
        if max_time - times[-1] < 0.1 and i == times.size:  # last loop
            lab = 'r'
        offlabels[i] = lab

    datFrame = pd.DataFrame({'times2': times, 'offtime': offtimes, 'side': offlabels})

    # Calculate the on times and assign labels
    ontimes = np.NaN*np.ones(times.size)
    onlabels = np.empty(times.size, dtype=str)
    if times[0] > exp_time:  # append the left kon if it exists
        ontimes[0] = times[0]
        onlabels[0] = 'l'

    for i in range(2, times.size, 2):
        ontimes[i-1] = times[i] - times[i-1]
        onlabels[i-1] = 'm'

    if max_time - times[-1] > exp_time:  # append the right kon if it exists
        ontimes[-1] = max_time - times[-1]
        onlabels[-1] = 'r'

    datFrame['ontime'] = ontimes
    datFrame['onside'] = onlabels
    datFrame['order'] = np.arange(0, times.size)

    # Check whether Ioffsets are the same for all molecule (w.r.t. to last)
    Icheck = int((mol.steps.Imin.tail(1) == mol.steps.Imin[0]) &
                 (mol.steps.Iroff.tail(1) == mol.steps.Iroff[0]) &
                 (mol.steps.Igoff.tail(1) == mol.steps.Igoff[0]))
    if Icheck == 0:  # check if thresholds the same for each dwell of the molecule
        print(f'Ioffsets are not equal to others for molecule:{i+1}')

    # Calculate the average FRET for each dwell
    fret = mol.E(Imin=mol.steps.Imin[0], Iroff=mol.steps.Iroff[0], Igoff=mol.steps.Igoff[0])
    avg_fret = np.NaN*np.ones(times.size)
    for ii in range(0, times.size, 2):
        istart = int(round(times[ii]/exp_time))
        iend = int(round(times[ii+1]/exp_time))
        a_fret = round(np.mean(fret[istart:iend]), 2)
        if (a_fret <= 1 and a_fret >= 0):
            avg_fret[ii] = a_fret
        else:
            print(f'FRET corrupted for molecule:{i+1}')

    datFrame['avrgFRET'] = avg_fret

    return datFrame


def process_molecule(mol):
    traces_unique = pd.unique(mol.steps.trace.values)
    results = []
    for trace in traces_unique:

        results.append(find_mol_dwells(mol, trace=trace))

    result = pd.concat(results, axis=0, ignore_index=True)
    mol.steps = pd.concat([mol.steps, result], axis=1)
    return mol.steps

def analyze_steps(file, save=True):
    file.time
    molecules_with_data = [mol for mol in file.molecules if mol.steps is not None]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(process_molecule, mol)
                   for mol in molecules_with_data]
    # if the results need to be manipulated uncomment the following lines:
    # for f in concurrent.futures.as_completed(results):
    #     print(f.result())

    filename = f'{file.relativeFilePath}_dwells_data.xlsx'
    data = file.savetoExcel(filename=filename, save=save)
    return data

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


