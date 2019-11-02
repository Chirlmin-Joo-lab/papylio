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
import traceAnalysisCode as analysis
sys.path.append('..')


def analyze_dwelltimes(exp_file, save=True, filename=None):
    exp_file.importExcel()  # this should not be needed normally
    max_time = exp_file.time[-1]
    exp_time = exp_file.exposure_time

    for i, mol in enumerate(exp_file.molecules):
        if mol.steps is None:
            continue
        times = mol.steps.time.sort_values().values
        try:
            times1 = times.reshape((int(times.size/2), 2))
        except ValueError:
            print(times)

#       Calculate the average FRET for each dwell
        avg_fret = []
        if 'exp_file.Imin' and 'exp_file.Iroff' in locals():
            fret = mol.E(Imin=exp_file.Imin, Iroff=exp_file.Iroff, Igoff=exp_file.Igoff)
        else:
            fret = mol.E()

        for ii in range(0, len(times)):
            if ii % 2 != 0:
                istart = int(round(times[ii-1]/exp_time))
                iend = int(round(times[ii]/exp_time))
                avg_fret.append(round(np.mean(fret[istart:iend]), 2))
            else:
                avg_fret.append('')
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
        data = exp_file.savetoExcel(filename=exp_file.name+'_dwells_data.xlsx')
    else:
        data = exp_file.savetoExcel(save=False)
    return data


if __name__ == '__main__':

    start = timetime.time()
#    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    expPath = 'O:/SM-data/20191024_dcas9_DNA05-06-09-Cy3'
    for chamberFolder in os.listdir(expPath):
        if 'movies' in chamberFolder:
            mainPath = expPath + f'/{chamberFolder}'
            exp = analysis.Experiment(mainPath)
            for file in exp.files:
                data = analyze_dwelltimes(file, save=True)


#    mainPath='./traces'
#    #mainPath = './iasonas/cas9_simulations/DNA11-20_30exposure'
#    exp = analysis.Experiment(mainPath)
#    for file.
#    file = exp.files[0]
#    data = analyze_dwelltimes(file, save=True)

#
    print(f'Analysis time: {timetime.time() - start} sec')
