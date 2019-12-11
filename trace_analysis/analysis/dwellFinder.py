# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:05:13 2019

@author: iason
"""
import time as timetime
import numpy as np
import sys
import pandas as pd
import os
#import matplotlib.pyplot as plt
import SAfitting.traceAnalysisCode as trace_ana
#from trace_analysis import Experiment
sys.path.append('..')


def find_dwelltimes(exp_file, trace='red', save=True, filename=None):
    #exp_file.importExcel()  # this should not be needed normally
    max_time = exp_file.time[-1]
    exp_time = exp_file.exposure_time

    for i, mol in enumerate(exp_file.molecules):
        if mol.steps is None or trace not in mol.steps.trace.values:
            continue
        times = mol.steps.time.sort_values().values[mol.steps.trace == trace]
        try:
            times1 = times.reshape((int(times.size/2), 2))
        except ValueError:
            print(times)
            return

#       Calculate the average FRET for each dwell
        avg_fret = []
        if 'Imin' and 'Iroff' in mol.steps.columns:
            Icheck = int((mol.steps.Imin.tail(1)== mol.steps.Imin[0])&(mol.steps.Iroff.tail(1) == mol.steps.Iroff[0])&(mol.steps.Igoff.tail(1) == mol.steps.Igoff[0]))
            if Icheck == 1:  #  check if thresholds the same for each dwell of the molecule
                fret = mol.E(Imin=mol.steps.Imin[0], Iroff=mol.steps.Iroff[0], Igoff=mol.steps.Igoff[0])
            else:
                print(f'Ioffsets are not equal for molecule:{i+1}')
                fret = []
        else:
            fret = mol.E()

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
            else:
                avg_fret.append(None)
        avgFRET = pd.DataFrame({'avg_FRET': avg_fret})

        dwells = np.diff(times1, axis=1).flatten()

        labels = []
        for i, d in enumerate(dwells):
            lab = 'm'
            if times[0] == 0 and i == 0:  # first loop
                lab = 'l'
            if max_time - times[-1] < 0.1 and i == len(dwells) - 1:  # last loop
                lab = 'r'
            labels.append(lab)
        dwells = pd.DataFrame({'offtime': dwells, 'side': labels})
#       Calculate the on times
        ontimes = []
        labels = []
        if times[0] != 0:  # append the left kon if it exists
            ontimes.append(times[0])
            labels.append('l')


#       for i, t in zip(range(1,times.size, 2), times[2:-1:2]):
        for i in range(2, times.size, 2):
            ontimes.append(times[i] - times[i-1])
            labels.append('m')

        if max_time - times[-1] > 0.1:  # append the right kon if it exists
            ontimes.append(max_time - times[-1])
            labels.append('r')

        ontimes = pd.DataFrame({'ontime': ontimes,
                                'onside': labels,
                                'order': np.arange(0, len(ontimes))})

        mol.steps = pd.concat([mol.steps, avgFRET, dwells, ontimes], axis=1)

    if save:
        data = exp_file.savetoExcel(filename=exp_file.name+f'_dwells_{trace}_data.xlsx')
    else:
        data = exp_file.savetoExcel(save=False)
    return data


if __name__ == '__main__':

    start = timetime.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    expPath = 'H:/SM-data/20191206_dcas9_flow_DNA20-03-08-07-05'
    for chamberFolder in os.listdir(expPath):
        if 'movies' in chamberFolder or 'movie' in chamberFolder:
            mainPath = expPath + f'/{chamberFolder}'
            exp = Experiment(mainPath)
            for file in exp.files:
                data = find_dwelltimes(file, save=True)


# #    mainPath='./traces'
#     mainPath = 'G:/SM-data/20191101_dcas9_flow_DNA04_DNA20/#4.20_streptavidin_0.5nM_dcas9-crRNA-Cy5_10nM_DNA04-Cy3_G_flow'
#     exp = Experiment(mainPath)
#     file = exp.files[1]

# #    data = find_dwelltimes(file, trace='red', save=True)

# #
#     print(f'Analysis time: {timetime.time() - start} sec')
