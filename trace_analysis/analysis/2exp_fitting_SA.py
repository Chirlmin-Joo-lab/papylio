# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:59:11 2019

@author: pimam
"""
import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="dark")
sns.set_color_codes()


def P(dwells, params):
    P1, tau1, tau2 = params
    return P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)


def LogLikeLihood(xdata, params, model):
    Pi = model(xdata, params)
    LLike = np.sum(-np.log(Pi))
    return LLike


def update_temp(T, alpha):
    '''
    Exponential cooling scheme
    :param T: current temperature.
    :param alpha: Cooling rate.
    :return: new temperature
    '''
    T *= alpha
    return T


def simmulated_annealing(data, objective_function, model, x_initial,
                         lwrbnd, uprbnd, Tstart=100., Tfinal=0.001,
                         delta1=0.1, delta2=2.5, alpha=0.9):

    T = Tstart
    step = 0
    x = x_initial
    while T > Tfinal:
        step += 1
        if (step % 100 == 0):
            T = update_temp(T, alpha)
        # print(x)
        x_trial = np.zeros(len(x))
        x_trial[0] = np.random.uniform(np.max([x[0] - delta1, lwrbnd[0]]),
                                       np.min([x[0] + delta1, uprbnd[0]]))
        for i in range(1, len(x)):
            x_trial[i] = np.random.uniform(np.max([x[i] - delta2, lwrbnd[i]]),
                                           np.min([x[i] + delta2, uprbnd[i]]))
#        x_trial = np.exp(x_trial)
#        x = np.exp(x)
        x = Metropolis(objective_function, model, x, x_trial, T, data)
    return x


def Metropolis(f, model, x, x_trial, T, data):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(data, x_trial, model)
    Vold = f(data, x, model)
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        x = x_trial
    return x


if __name__ == '__main__':


#    # Import data and prepare for fitting

    filename = 'H:/SM-data/20191101_dcas9_flow_DNA04_DNA20/'
    chamber = '#5.10_streptavidin_0.5nM_biot-dcas9-Cy5_10nM_DNA05-Cy3_G_movies_flow'
    filename += chamber + '/' + 'hel9_dwells_red_data.xlsx'

    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})


    dwelltype = 'offtime'

    dwells = data[dwelltype].values
    dwells = dwells[~np.isnan(dwells)]

    # dwells_rec = dwells[dwells < dwells.max() - 5]
    # dwells_cut = dwells[dwells >= dwells.max() - 5]


    # Set parameters for simmulated annealing
    N = 5  # number of fits performed
    max_dwells = dwells.max()
    avg_dwells = np.average(dwells)
    x_initial = [0.5, avg_dwells, avg_dwells]
    lwrbnd = [0, 0, 0]
    uprbnd = [1, max_dwells, max_dwells]

    # Perform N fits on data using simmulated annealing
    fitdata = simmulated_annealing(data=dwells, objective_function=LogLikeLihood, model=P, x_initial=x_initial, lwrbnd=lwrbnd, uprbnd=uprbnd)
    print("fit found: ", str(fitdata))
    fitparams = [fitdata]
    for i in range(1, N):
        fitdata = simmulated_annealing(data=dwells, objective_function=LogLikeLihood, model=P, x_initial=x_initial, lwrbnd=lwrbnd, uprbnd=uprbnd)
        print("fit found: ", str(fitdata))
        fitparams = np.concatenate((fitparams, [fitdata]), axis=0)

    # Plot the dwell time histogram and the corresponding fits
    plt.figure()
    values, bins = np.histogram(dwells, bins=10, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, 'r.', label=f'offtimes N={dwells.size}')



    LLike = np.zeros(N)
    timearray = np.linspace(0, max_dwells, num=1000)
    for i in range(0, np.size(fitparams, 0)):
        fit = P(timearray, fitparams[i])
        LLike[i] = LogLikeLihood(dwells, fitparams[i], P)
        # plt.plot(timearray, fit, label='fit'+str(i))
    iMaxLike = np.argmax(LLike)
    bestparams = fitparams[iMaxLike]
    bestfit = P(timearray, fitparams[iMaxLike])
    plt.plot(timearray, bestfit, 'b', label='p: '+"{0:.2f}".format(fitparams[iMaxLike][0])+"\n"+r'$\tau_1$: '+"{0:.1f}".format(fitparams[iMaxLike][1])+"\n"+r'$\tau_2$: '+"{0:.1f}".format(fitparams[iMaxLike][2]))


    # plot single exponential fit
    exp = 1/avg_dwells*np.exp(-timearray/avg_dwells)
    plt.plot(timearray, exp, 'orange', label = rf'$\tau$: {avg_dwells: .1f}')

    plt.xlabel('dwell time (sec)')
    plt.ylabel('prob. density')
    plt.legend(fontsize='x-large')
    plt.savefig(filename[:-5] +'_offtime_dist_fit.png', dpi=200)

