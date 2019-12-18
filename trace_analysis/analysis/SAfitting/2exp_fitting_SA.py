# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:59:11 2019

@author: pimam
"""
if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path, PureWindowsPath
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    p = Path(__file__).parents[3]
    sys.path.insert(0, str(p))
    mainPath = PureWindowsPath('F:\\20191211_dCas9_DNA5_7_8_20_Vikttracr\\#3_strept_1nMCas9_10nMDNA7_movies\\peaks5collect4\\analysis_green_red')

from trace_analysis import Experiment
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="dark")
sns.set_color_codes()


def P2expcut(dwells, params, Tcut):
    P1, tau1, tau2 = params
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
    return Pi, Pcut


def LogLikeLihood(xdata, params, model, Tcut, Ncut):
    Pi, Pcut = model(xdata, params, Tcut)
    LLikecut = 0
    if Ncut != 0:
        LLikecut = -Ncut * np.log(Pcut)
    LLike = np.sum(-np.log(Pi)) + LLikecut
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


def simmulated_annealing(data, objective_function, model, x_initial, lwrbnd,
                         uprbnd, Tcut, Ncut, Tstart=100.,
                         Tfinal=0.001, delta1=0.1, delta2=2.5, alpha=0.9):
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
        x = Metropolis(objective_function, model, x, x_trial, T, data,
                       Tcut, Ncut)
    return x


def Metropolis(f, model, x, x_trial, T, data, Tcut, Ncut):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(data, x_trial, model, Tcut, Ncut)
    Vold = f(data, x, model, Tcut, Ncut)
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        x = x_trial
    return x


def Nfits_sim_anneal(dwells, N, model, Tcut=0, Ncut=0):
    # Perform N fits on data using simmulated annealing
    for i in range(0, N):
        fitdata = simmulated_annealing(data=dwells,
                                       objective_function=LogLikeLihood,
                                       model=model, x_initial=x_initial,
                                       lwrbnd=lwrbnd, uprbnd=uprbnd,
                                       Tcut=Tcut, Ncut=Ncut)
        print("fit found: ", str(fitdata))
        if i == 0:
            fitparams = [fitdata]
        else:
            fitparams = np.concatenate((fitparams, [fitdata]), axis=0)
    return fitparams


if __name__ == '__main__':

    # Import data and prepare for fitting
    exp = Experiment(mainPath)
    file = exp.files[0]
    filename = './'+file.name+'_dwells_data.xlsx'
    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon': np.str})

    if len(exp.files) > 1:  # time of traces should be of the same length
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
    Ntot = len(dwells_all)
    Tmax = dwells_all.max()
    Tcut = Tmax - 10
    dwells_rec = dwells_all[dwells_all < Tcut]
    Ncut = np.size(dwells_all[dwells_all >= Tcut])
    print('Ntot ', Ntot)
    print('Tcut ', Tcut)
    print('Ncut ', Ncut)
    dwells = dwells_rec

#    filename = '2exp1_N=10000_rep=1_tau1=10_tau2=100_a=0.5'
#    dwells_all = np.load('./data/2exp1_N=10000_rep=1_tau1=10_tau2=100_a=0.5.npy')
#    max_alldwells = dwells_all.max()
#    Tcut = max_alldwells - 20
#    dwells = dwells_all[dwells_all < Tcut]
#    Ncut = np.sum(dwells_all >= Tcut) + 40
#    print('Ncut: ', Ncut)

    # Set parameters for simmulated annealing
    N = 10
    model = P2expcut
    avg_dwells = np.average(dwells)
    x_initial = [0.5, avg_dwells, avg_dwells]
    lwrbnd = [0, 0, 0]
    uprbnd = [1, 1.5*Tmax, 1.5*Tmax]

    # Perform N fits on data using simmulated annealing
    # If you want to use P2exp without Ncut, just fil in Ncut=0 and Tcut=0
    fitparams = Nfits_sim_anneal(dwells, N, model=model, Tcut=Tcut, Ncut=Tcut)

    # Plot the dwell time histogram and the corresponding fits
    plt.figure()
    values, bins = np.histogram(dwells, bins=10, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, 'r.', label=f'offtimes N={dwells.size}')

    LLike = np.zeros(N)
    timearray = np.linspace(0, Tmax, num=1000)
    for i in range(0, np.size(fitparams, 0)):
        fit, Pcut = model(timearray, fitparams[i], Tcut)
        LLike[i] = LogLikeLihood(dwells, fitparams[i], model, Tcut, Ncut)
        plt.plot(timearray, fit, label='fit'+str(i))

    # Find best fit, plot with histogram and save
    plt.figure()
    values, bins = np.histogram(dwells, bins=40, density=True)
    centers = (bins[1:] + bins[:-1]) / 2.0
    plt.plot(centers, values, '.', label=f'Dwells with Ncut:{Ncut}')

    iMaxLike = np.argmax(LLike)
    bestparams = fitparams[iMaxLike]
    bestfit, Pcutbest = model(timearray, fitparams[iMaxLike], Tcut)
    plt.plot(timearray, bestfit, label='P1:'+"{0:.2f}".format(fitparams[iMaxLike][0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(fitparams[iMaxLike][1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(fitparams[iMaxLike][2]))
    plt.xlabel('dwell time (sec)')
    plt.ylabel('prob. density')
    plt.legend(fontsize='x-large')
    plt.savefig(f'{len(exp.files)}files_bestfit.png', facecolor='white', dpi=200)

    # Plot data with double and single exponential fit
    plt.figure()
    plt.semilogy(centers, values, '.', label=f'Dwells with Ncut:{Ncut}')
    plt.semilogy(timearray, bestfit, label='P1:'+"{0:.2f}".format(fitparams[iMaxLike][0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(fitparams[iMaxLike][1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(fitparams[iMaxLike][2]))
    singlexp = 1/avg_dwells*np.exp(-timearray/avg_dwells)
    plt.plot(timearray, singlexp, 'orange', label = rf'$\tau$:{avg_dwells:.1f}')
    plt.xlabel('dwell time (sec)')
    plt.ylabel('log prob. density')
    plt.legend(fontsize='x-large')
    plt.savefig(f'{len(exp.files)}files_1_2expfit__compared.png', dpi=200)
