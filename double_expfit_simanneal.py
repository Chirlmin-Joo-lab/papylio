# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:35:27 2019

@author: pimam
"""

import os
import traceAnalysisCode as trace_ana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath
import SimulatedAnnealing_parallel_dblexp as SA

def P(dwells, params):
    P1, t1, t2 = params
    return P1/t1*np.exp(-dwells/t1)+(1-P1)/t2*np.exp(-dwells/t2)

Path = 'C:\\Users\\pimam\\Documents\\MEP\\tracesfiles'
mainPath = PureWindowsPath(Path)

os.remove(Path + '\\init_monitor.txt')
os.remove(Path + '\\monitor.txt')
#except IOError:
#    print('monitor file not present')

exp = trace_ana.Experiment(mainPath)



file = exp.files[0]
filename = './'+file.name+'_dwells_data.xlsx'
data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})

if len(exp.files) > 1:  #   time of traces should be of the same length  
    for file in exp.files[1:]:
        filename = './'+file.name+'_dwells_data.xlsx'
        print(filename)
        data2 = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})
        data = data.append(data2, ignore_index=True)

dwelltype = 'offtime'
dwells_all = []
dwells = data[dwelltype].values
dwells = dwells[~np.isnan(dwells)]
dwells_all.append(dwells)
dwells_all = np.concatenate(dwells_all)
avg_dwells = np.average(dwells_all)
max_dwells = dwells_all.max()

dwells_rec = dwells[dwells < max_dwells - 10]
dwells_cut = dwells[dwells >= max_dwells - 10]
Earray = np.zeros(len(dwells_rec))

plt.figure()
values, bins = np.histogram(dwells_rec, bins=100, density=True)
centers = (bins[1:] + bins[:-1]) / 2.0
plt.plot(centers, values, '.', label='All dwells')

fitparams = [[0.2246899, 5.69473985, 99.25577865],[0.23903558, 4.60093276, 93.53687637], [0.79048165, 92.17272163, 4.21477293], [0.22695878,  4.85038366, 94.18223422], [0.75890422, 93.53996164,  6.09512603]]
timearray = np.linspace(0, max_dwells, num=200)

for i in range(0, np.size(fitparams, 0)):
    fit = P(timearray, fitparams[i])
    plt.plot(timearray, fit, label='fit'+str(i))


fitdata = SA.sim_anneal_fit(xdata=dwells_rec, ydata=Earray, yerr=Earray, Xstart=[0.5, avg_dwells, avg_dwells], lwrbnd=[0.001, 1, 1], upbnd=[1, max_dwells, max_dwells],
                            model=P, objective_function='Maximum Likelihood', delta=0.05, delta2=1.0, tol=1, cooling_rate=0.04, use_relative_steps=False)

fitparams = np.concatenate((fitparams, [fitdata]), axis=0)

fit = P(timearray, fitdata)
plt.plot(timearray, fit, label='new fit')
plt.legend()