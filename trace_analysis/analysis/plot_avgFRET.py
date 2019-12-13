# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:09:03 2019

@author: pimam
"""
if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path, PureWindowsPath
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = Path(__file__).parents[2]
    sys.path.insert(0, str(p))
    mainPath = PureWindowsPath('F:\\20191211_dCas9_DNA5_7_8_20_Vikttracr\\#5_strept_1nMCas9_10nMDNA20_snaps\\peaks5collect4_green_red\\histograms')   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trace_analysis import Experiment, plotting
import seaborn as sns
sns.set(style="ticks")
sns.set_color_codes()

def plot_avgFRET(exp, bins=100, save=True):
    
    Edata=[]
    for fl in exp.files:
        nmole= len(fl.molecules)
        print(nmole)
        data = np.array([np.mean(molecule.E()) for molecule in fl.molecules])
        Edata = np.concatenate((Edata, data), axis=0)
        plt.figure()
        plt.hist(data, bins, range = (0,1))
        plt.title(f'avgFRET {fl.name} bins:{bins} mole:{nmole}') 
        plt.xlabel('FRET')
        plt.ylabel('count')
        plt.savefig(f'{fl.name}_avgFRET_hist.png', facecolor='white', dpi=200)
    nmole = np.size(Edata)
    plt.figure()
    plt.hist(data, bins, range = (0,1))
    plt.title(f'avgFRET all snaps bins:{bins} mole:{nmole}')
    plt.xlabel('FRET')
    plt.ylabel('count')
    plt.savefig('avgFRETall_hist.png', facecolor='white', dpi=300)

if __name__ == '__main__':
    exp = Experiment(mainPath)
    bins=100
    plot_avgFRET(exp, bins, save=True)

