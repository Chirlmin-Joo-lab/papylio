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
import traceAnalysisCode as trace_ana


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


def simmulated_annealing(data, objective_function, model, x_initial, lwrbnd, uprbnd, Tstart=100., Tfinal=0.001, delta1=0.1, delta2=2.5, alpha=0.9):

    T = Tstart
    step = 0
    x = x_initial
    while T > Tfinal:
        step += 1
        if (step % 100 == 0):
            T = update_temp(T, alpha)
        print(x)
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
    try:
        os.remove('./init_monitor.txt')
        os.remove('./monitor.txt')
    except IOError:
        print('monitor file not present')

    # Import data and prepare for fitting
#    Path = 'C:\\Users\\pimam\\Documents\\MEP\\traces1'
#    mainPath = PureWindowsPath(Path)
#    exp = trace_ana.Experiment(mainPath)
#    file = exp.files[0]
#    filename = './'+file.name+'_dwells_data.xlsx'
#    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})
#
#    if len(exp.files) > 1:  # time of traces should be of the same length
#        for file in exp.files[1:]:
#            filename = './'+file.name+'_dwells_data.xlsx'
#            print(filename)
#            data2 = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})
#            data = data.append(data2, ignore_index=True)
#
#    dwelltype = 'offtime'
#    dwells_all = []
#    dwells = data[dwelltype].values
#    dwells = dwells[~np.isnan(dwells)]
#    dwells_all.append(dwells)
#    dwells_all = np.concatenate(dwells_all)
#    max_alldwells = dwells_all.max()
#    dwells_rec = dwells[dwells < max_alldwells - 10]
#    dwells_cut = dwells[dwells >= max_alldwells - 10]

    dwells_rec = np.load('./data/2exp1_N=10000_rep=1_tau1=10_tau2=100_a=0.5.npy')
    dwells_cut = []

    # Set parameters for simmulated annealing
    N = 10
    max_dwells = dwells_rec.max()
    avg_dwells = np.average(dwells_rec)
    x_initial = [0.5, avg_dwells, avg_dwells]
    lwrbnd = [0, 0, 0]
    uprbnd = [1, max_dwells, max_dwells]

    # Perform N fits on data using simmulated annealing
    fitdata = simmulated_annealing(data=dwells_rec, objective_function=LogLikeLihood, model=P, x_initial=x_initial, lwrbnd=lwrbnd, uprbnd=uprbnd)
    print("fit found: ", str(fitdata))
    fitparams = [fitdata]
    for i in range(1, N):
        fitdata = simmulated_annealing(data=dwells_rec, objective_function=LogLikeLihood, model=P, x_initial=x_initial, lwrbnd=lwrbnd, uprbnd=uprbnd)
        print("fit found: ", str(fitdata))
        fitparams = np.concatenate((fitparams, [fitdata]), axis=0)

    # Plot the dwell time histogram and the corresponding fits
    plt.figure()
    values, bins = np.histogram(dwells_rec, bins=20, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, '.', label='All dwells')

    LLike = np.zeros(N)
    timearray = np.linspace(0, max_dwells, num=200)
    for i in range(0, np.size(fitparams, 0)):
        fit = P(timearray, fitparams[i])
        LLike[i] = LogLikeLihood(dwells_rec, fitparams[i], P)
        plt.plot(timearray, fit, label='fit'+str(i))

    # Find best fit, plot with histogram and save
    plt.figure()
    values, bins = np.histogram(dwells_rec, bins=60, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, '.', label='All dwells')

    iMaxLike = np.argmax(LLike[1:])
    bestparams = fitparams[iMaxLike]
    bestfit = P(timearray, fitparams[iMaxLike])
    plt.plot(timearray, bestfit, label='P1:'+"{0:.2f}".format(fitparams[iMaxLike][0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(fitparams[iMaxLike][1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(fitparams[iMaxLike][2]))
    plt.xlabel('dwell time (sec)')
    plt.ylabel('fraction')
    plt.legend()
    #plt.savefig(f'{len(exp.files)}files_bestfit.png', facecolor='white', dpi=300)

    plt.figure()
    plt.semilogy(centers, values, '.', label='All dwells')
    plt.semilogy(timearray, bestfit, label='P1:'+"{0:.2f}".format(fitparams[iMaxLike][0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(fitparams[iMaxLike][1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(fitparams[iMaxLike][2]))
    plt.xlabel('dwell time (sec)')
    plt.ylabel('logfraction')
    plt.legend()
    #plt.savefig(f'{len(exp.files)}files_bestfit_log.png', facecolor='white', dpi=300)
