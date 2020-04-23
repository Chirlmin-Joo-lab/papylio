# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:41:20 2020

@author: iason
"""

if __name__ == '__main__':
    import sys
    from pathlib import Path
    p = Path(__file__).parents[2]  # include two levels up directory where the trace_analysis package lies
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from trace_analysis import Experiment

import matplotlib.pyplot as plt
import seaborn as sns

def plot_fret_histogram(molecules, Emin=0.05, Emax=0.8, Irmin=40, Iroff=0,
                        Igoff=0, **hist_kwargs):
    plt.style.use('seaborn-dark')
    plt.style.use('seaborn-colorblind')
    data = np.concatenate([molecule.E(Irmin, Iroff, Igoff) for molecule in molecules])
    data = data[data>Emin]
    data = data[data<Emax]
    centers, bins, _ = plt.hist(data, **hist_kwargs)


def get_fret_for_offtimes(molecules, trace_type='red', Emin=0.05, Emax=1,
                        Irmin=40, Iroff=0, Igoff=0, fret_max=False):
    '''
    Function that gives the FRET trace with FRET calculcated only for the
    selected offtime interval. The FRET value outside this interval is put to
    zero
    '''
    fret_new = []
    for mol in list(molecules):
        if mol.steps is None:
            continue
        print(mol.index)
        time_axis = mol.file.time
        # select the times in which steps are taken for the particular trace_type
        times = mol.steps.time.values[mol.steps.trace == trace_type]
        indices = np.empty(times.size, dtype=int)
        for i, t in enumerate(times):
            idx = find_nearest(time_axis, t)
            indices[i] = idx
        fret = mol.E(Irmin, Iroff, Igoff)
        idx_start = indices[::2]
        idx_stop = indices[1::2]

        if fret_max:
            fret_reduced = []
            for i in range(idx_start.size):
                try:
                    maximum = np.max(fret[idx_start[i]+1:idx_stop[i]])
                except ValueError:
                    pass
                fret_new.append(maximum)

        else:
            mask = np.zeros(fret.size, dtype=int)
            for i in range(idx_start.size):
                mask[idx_start[i]+1:idx_stop[i]] = 1
            # Multiply by zero all the fret values outside the offtime intervals
            fret_reduced = np.multiply(fret, mask)

            fret_new.append(fret_reduced)

    fret_new = np.array(fret_new)
    fret_new = fret_new[fret_new > 0]

    return fret_new


def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


if __name__ == '__main__':

    mainPath = r'C:/Users/iason/Desktop/traceanalysis/trace_analysis/traces/test_data/DNA08'

    exp = Experiment(mainPath)
    file = exp.files[0]
    filename = mainPath + '/hel21_dwells_data.xlsx'
    dwell_data =  pd.read_excel(filename, index_col=[0, 1], dtype={'kon' :np.str})
    # fret_hist = plot_fret_histogram(file.molecules, bins=100)
    mol = file.molecules[0]
    fret = get_fret_for_offtimes(file.molecules)
    plt.hist(fret, bins=50)

