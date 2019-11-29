# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:09:03 2019

@author: pimam
"""

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath
import traceAnalysisCode as trace_ana

#mainPath = PureWindowsPath('C:\\Users\\pimam\\Documents\\MEP\\tracesfiles')
mainPath = PureWindowsPath('F:\\20191127_wtCas9\\#2_strept_100pMDNA20biot_10nMwtCas9_1to2_snaps\\peaks5collect4_red_green')
mainPath = Path(mainPath)
exp = trace_ana.Experiment(mainPath)
k=0
nbins=100

for i in range(0,len(exp.files)):
    file = exp.files[i]
    nmole= len(file.molecules)
    k+=1
    plt.figure(k)
    trace_ana.histogram(file.molecules, nbins, None)
    plt.title(f'FRET histogram nbins:{nbins} mole:{nmole}') 
    plt.xlabel('FRET')
    plt.ylabel('count')
    plt.savefig(f'{file.name}_FRET_hist.png', facecolor='white', dpi=300)
    k+=1
    plt.figure(k)
    data = [np.mean(molecule.E()) for molecule in file.molecules]
    plt.hist(data, nbins, range = (0,1))
    plt.title(f'avgFRET histogram nbins:{nbins} mole:{nmole}') 
    plt.xlabel('FRET')
    plt.ylabel('count')
    plt.savefig(f'{file.name}_avgFRET_hist.png', facecolor='white', dpi=300)

data=[]
for fl in exp.files:
    dataE=np.array([np.mean(molecule.E()) for molecule in fl.molecules])
    data = np.concatenate((data, dataE), axis=0)
    szdata=len(data)
    print(szdata)
k+=1
plt.figure(k)
plt.hist(data, nbins, range = (0,1))
plt.title(f'avgFRET histogram all snaps nbins:{nbins} mole:{szdata}') 
plt.xlabel('FRET')
plt.ylabel('count')
plt.savefig('avgFRETall_hist.png', facecolor='white', dpi=300)