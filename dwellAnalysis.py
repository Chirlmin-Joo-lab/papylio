# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:32:58 2019

@author: ikatechis
"""

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.set_color_codes()

import traceAnalysisCode as analysis

def get_dwell_hist(dwells, dwelltype='offtime', nbins=20, save=True, plot=True, extra_label='', log=False):

    #  select only the ones that don't exceed the total measurement time minus 10 sec
    dwells_in = dwells[dwells < dwells.max() - 10]
    avrg_dwell = get_average_dwell(dwells)



    values, bins = np.histogram(dwells, bins=nbins, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    if not plot:
        return values, centers

    if plot:
        line = plt.plot(centers, values, '.', label=extra_label+fr'$\tau = ${avrg_dwell:.1f} s' )[0]

        plt.xlabel('time (s)')
        plt.ylabel('Prob.')
        plt.title(f'{dwelltype} histogram: nbins={nbins} N={dwells.size}')
        plt.legend(prop={'size': 16})
        # plot a 1exp ML 'fit' for the average dwelltime
        t = np.arange(0, dwells.max(), 0.1)
        exp = 1/avrg_dwell*np.exp(-t/avrg_dwell)
        if log:
            plt.semilogy(t, exp, color=line.get_color())
        else:
            plt.plot(t, exp, color=line.get_color())


        return line

def get_average_dwell(dwells):
    #  correct for dwell exceeding the measurement time
    Tmax = dwells.max() - 10
    Ntot = dwells.size
    Ncut = dwells[dwells > Tmax].size
    avrg_dwell = np.average(dwells[dwells < dwells.max() - 10])
    avrg_dwell = avrg_dwell + Ncut*Tmax/Ntot
    return avrg_dwell

if __name__ == '__main__':
    filename = 'O:/SM-data/20191024_dcas9_DNA05-06-09-Cy3/'
    chamber = '#4.20_streptadivin_0.5nM_dcas9-crRNA-Cy5_10nM_DNA05-Cy3_G_movies_lowsalt'
    filename += chamber + '/'
    dwells_all = []
    for path in os.listdir(filename):
        if 'dwells_data' in path:

            data = pd.read_excel(filename+path, index_col=[0, 1], dtype={'kon' :np.str})
            dwells = data['offtime'].values
            dwells = dwells[~np.isnan(dwells)]
            hist = get_dwell_hist(dwells)
            dwells_all.append(dwells)

    dwells_all = np.concatenate(dwells_all)

#    hist = get_dwell_hist(dwells_all, extra_label='All: ')
#    plt.savefig(filename+'dwelltime_dist.png', dpi=200)

    hist = get_dwell_hist(dwells_all, extra_label='All: ', log=True)
    plt.savefig(filename+'dwelltime_dist_log.png', dpi=200)