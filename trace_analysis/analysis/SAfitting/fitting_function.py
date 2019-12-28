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
    mainPath = PureWindowsPath('C:\\Users\\pimam\\Documents\\MEP\\tracesfiles')

from trace_analysis import Experiment
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="dark")
sns.set_color_codes()


def ML1expcut(dwells, Tcut, Ncut):
    Nrec = dwells.size
    if Ncut != 0:
        avg_dwells = np.average(dwells[dwells < Tcut])
    else:
        avg_dwells = np.average(dwells)
    MLtau = avg_dwells + Ncut*Tcut/Nrec
    timearray = np.linspace(0, Tcut, 1000)
    P = 1/MLtau*np.exp(-timearray/MLtau)
    return P, MLtau


def ML2expcut(dwells, params, Tcut, Ncut):
    P1, tau1, tau2 = params
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    LLike = np.sum(-np.log(Pi))
    if Ncut != 0:
        Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
        LLikecut = -Ncut * np.log(Pcut)
        LLike += LLikecut
    return Pi, LLike


def P2expcut(dwells, params, Tcut, Ncut):
    P1, tau1, tau2 = params
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
    return Pi, Pcut


def LogLikeLihood(xdata, params, model, Tcut, Ncut):
    Pi, Pcut = model(xdata, params, Tcut, Ncut)
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
    i = 0
    T = Tstart
    step = 0
    xstep = 0
    x = x_initial
    while T > Tfinal:
        step += 1
        if (step % 100 == 0):
            T = update_temp(T, alpha)
        x_trial = np.zeros(len(x))
        x_trial[0] = np.random.uniform(np.max([x[0] - delta1, lwrbnd[0]]),
                                       np.min([x[0] + delta1, uprbnd[0]]))
        for i in range(1, len(x)):
            x_trial[i] = np.random.uniform(np.max([x[i] - delta2, lwrbnd[i]]),
                                           np.min([x[i] + delta2, uprbnd[i]]))
        x, xstep = Metropolis(objective_function, model, x, x_trial, T, data,
                              Tcut, Ncut, xstep)
    print(f'steps: {xstep}')
    return x, xstep


def Metropolis(f, model, x, x_trial, T, data, Tcut, Ncut, xstep):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(data, x_trial, model, Tcut, Ncut)
    Vold = f(data, x, model, Tcut, Ncut)
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        x = x_trial
        xstep += 1
    return x, xstep


def Nfits_sim_anneal(dwells, Nfits, model, x_initial,
                     lwrbnd, uprbnd, Tcut, Ncut):
    # Perform N fits on data using simmulated annealing
    LLike = np.empty(Nfits)
    for i in range(0, Nfits):
        fitdata, xstep = simmulated_annealing(data=dwells,
                                              objective_function=LogLikeLihood,
                                              model=model, x_initial=x_initial,
                                              lwrbnd=lwrbnd, uprbnd=uprbnd,
                                              Tcut=Tcut, Ncut=Ncut)
        print(f"fit{i} found: {fitdata}")
        if i == 0:
            fitparam = [fitdata]
            Nsteps = [xstep]
        else:
            fitparam = np.concatenate((fitparam, [fitdata]), axis=0)
            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)
            LLike[i] = LogLikeLihood(dwells, fitparam[i], model, Tcut, Ncut)
    Ncutarray = Ncut*np.ones(len(Nsteps))
    ibestparam = np.argmax(LLike)
    return fitparam, Nsteps, Ncutarray, ibestparam


def Bootstrap_sim_anneal(dwells, repeats, model, x_initial,
                         lwrbnd, uprbnd, Tcut, Ncut):
    LLike = np.empty(repeats)
    Ncutarray = np.empty(repeats)
    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
    print(f'dwells size: {dwells_Ncut.size}')
    for i in range(0, repeats):
        dwells_rand = np.random.choice(dwells_Ncut, 1000)
        dwells = dwells_rand[dwells_rand > 0]
        Ncut = np.count_nonzero(dwells_rand == 0)
        Ncutarray[i] = Ncut
        fitdata, xstep = simmulated_annealing(data=dwells,
                                              objective_function=LogLikeLihood,
                                              model=model, x_initial=x_initial,
                                              lwrbnd=lwrbnd, uprbnd=uprbnd,
                                              Tcut=Tcut, Ncut=Ncut)
        print(f"fit{i} found: {fitdata} Ncut: {Ncut}")
        if i == 0:
            fitparam = [fitdata]
            Nsteps = [xstep]
        else:
            fitparam = np.concatenate((fitparam, [fitdata]), axis=0)
            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)
            LLike[i] = LogLikeLihood(dwells, fitparam[i], model, Tcut, Ncut)
    ibestparam = np.argmax(LLike)
    return fitparam, Nsteps, Ncutarray, ibestparam


def fitting(dwells_all, mdl, Nfits, include_over_Tmax=True,
            bootstrap=False, boot_repeats=0):
    Tmax = dwells_all.max()
    if include_over_Tmax is True:
        Tcut = 300  # Tmax - 10
        dwells = dwells_all[dwells_all < Tcut]
        Ncut = dwells_all[dwells_all >= Tcut].size
    else:
        Tcut = 0
        Ncut = 0
        dwells = dwells_all
    print(f'Ncut: {Ncut}')

    if mdl == '1Exp':
        if Nfits > 1:
            print('Multiple fit not applicable for 1Exp fitting (Nfits>1)')
        if bootstrap is True:
            print('Bootstrap not applicable for 1Exp fitting')

        model = ML1expcut
        fit, bestparam = ML1expcut(dwells, Tcut, Ncut)

        # Save data of interest to dataframe
        data = pd.DataFrame({'P1': [100], 'tau1': [bestparam], 'tau2': ['Nan'],
                             'Nsteps': ['Nan'], 'Ncut': [Ncut]})

        # Simple plot of fit to the histogram
        plt.figure()
        timearray = np.linspace(0, Tmax, 1000)
        values, bins = np.histogram(dwells, bins=40, density=True)
        centers = (bins[1:] + bins[:-1]) / 2.0
        plt.semilogy(centers, values, '.', label=f'Dwells')
        plt.semilogy(timearray, fit, label=rf'$\tau$ML:{bestparam:.1f}')
        plt.xlabel('dwell time (sec)')
        plt.ylabel('log prob. density')
        plt.legend()
    elif mdl == '2Exp':
        model = P2expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = [0.5, avg_dwells, avg_dwells]
        lwrbnd = [0, 0, 0]
        uprbnd = [1, 2*Tmax, 2*Tmax]

        if bootstrap is True:
            if Nfits > 1:
                print('Nfits is ignored, because Bootstrapping is used')
            fitparam, Nsteps, Ncutarray, ibestparam = Bootstrap_sim_anneal(
                                                       dwells, boot_repeats,
                                                       model=model,
                                                       x_initial=x_initial,
                                                       lwrbnd=lwrbnd,
                                                       uprbnd=uprbnd,
                                                       Tcut=Tcut,
                                                       Ncut=Ncut)
        else:
            # Perform N fits on data using simmulated annealing
            fitparam, Nsteps, Ncutarray, ibestparam = Nfits_sim_anneal(
                                                       dwells, Nfits,
                                                       model=model,
                                                       x_initial=x_initial,
                                                       lwrbnd=lwrbnd,
                                                       uprbnd=uprbnd,
                                                       Tcut=Tcut,
                                                       Ncut=Ncut)

        # Save data of interest to dataframe
        bestparam = fitparam[ibestparam]
        bestNsteps = Nsteps[ibestparam]
        bestNcut = Ncutarray[ibestparam]
        Allfitparam = np.concatenate((fitparam, [bestparam]), axis=0)
        Nsteps = np.concatenate((Nsteps, [bestNsteps]), axis=0)
        Ncutarray = np.concatenate((Ncutarray, [bestNcut]), axis=0)
        data = pd.DataFrame(Allfitparam)
        data.columns = ['P1', 'tau1', 'tau2']
        data['Nsteps'] = Nsteps
        data['Ncut'] = Ncutarray
        idx = []
        for i in range(len(fitparam)):
            idx.append('fit' + str(i+1))
        idx.append('Bestfit')
        data.index = idx

        # Quick plot of the dwell time histogram and the corresponding fits
        plt.figure()
        values, bins = np.histogram(dwells, bins=40, density=True)
        centers = (bins[1:] + bins[:-1]) / 2.0
        plt.plot(centers, values, 'r.', label=f'offtimes N={dwells.size}')
        timearray = np.linspace(0, Tmax, num=1000)
        for i in range(0, np.size(fitparam, 0)):
            fit, Pcut = model(timearray, fitparam[i], Tcut, Ncut)
            plt.plot(timearray, fit, label='fit'+str(i+1))
        plt.xlabel('dwell time (sec)')
        plt.ylabel('prob. density')

        # Quick plot of best fit with histogram
        plt.figure()
        plt.title('Best double exponential fit found')
        values, bins = np.histogram(dwells, bins=40, density=True)
        centers = (bins[1:] + bins[:-1]) / 2.0
        plt.semilogy(centers, values, '.', label=f'Dwells')
        bestfit, Pcutbest = model(timearray, bestparam, Tcut, Ncut)
        plt.semilogy(timearray, bestfit, label='P1:'+"{0:.2f}".format(bestparam[0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(bestparam[1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(bestparam[2]))
        plt.xlabel('dwell time (sec)')
        plt.ylabel('log prob. density')
        plt.legend()

    return data


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

#    filename = '2exp1_N=10000_rep=1_tau1=10_tau2=100_a=0.5'
#    dwells_all = np.load('./data/2exp1_N=10000_rep=1_tau1=10_tau2=100_a=0.5.npy')

    # Start fitting
    mdl = '2Exp'
    include_over_Tmax = True
    Nfits = 2
    bootstrap = True
    boot_repeats = 200
    fitdata = fitting(dwells_all, mdl, Nfits, include_over_Tmax, bootstrap, boot_repeats)
    print(fitdata)
    fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv', index=False)
    newdata = pd.read_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv')
    # Getting measures and plotting the parameter values found
    taubnd = 100
    fitP1 = []
    fittau1 = []
    fittau2 = []
    for i in range(0, len(fitdata['tau1'])):
        if fitdata['tau1'][i] > taubnd:
            fittau2.append(fitdata['tau1'][i])
            fitP1.append(1-fitdata['P1'][i])
        else:
            fittau1.append(fitdata['tau1'][i])
            fitP1.append(fitdata['P1'][i])
        if fitdata['tau2'][i] > taubnd:
            fittau2.append(fitdata['tau2'][i])
        else:
            fittau1.append(fitdata['tau2'][i])

    P1_avg = np.average(fitP1)
    tau1_avg = np.average(fittau1)
    tau2_avg = np.average(fittau2)
    P1_std = np.std(fitP1)
    tau1_std = np.std(fittau1)
    tau2_std = np.std(fittau2)
    Nbins = 50

    plt.figure()
    plt.hist(fitP1, bins=Nbins)
    plt.vlines(P1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(P1_avg))
    plt.title(f'Fit values for P1 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()
    plt.figure()
    plt.hist(fittau1, bins=Nbins)
    plt.vlines(tau1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau1_avg))
    plt.title(rf'Fit values for $\tau$1 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()
    plt.figure()
    plt.hist(fittau2, bins=Nbins)
    plt.vlines(tau2_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau2_avg))
    plt.title(rf'Fit values for $\tau$1 Nfits: {boot_repeats} Nbins: {Nbins}')
    plt.legend()


#    # Plot data with double and single exponential fit
#    plt.figure()
#    plt.semilogy(centers, values, '.', label=f'Dwells with Ncut:{Ncut}')
#    plt.semilogy(timearray, bestfit, label='P1:'+"{0:.2f}".format(bestparams[0])+"\n"+r'$\tau$1:'+"{0:.1f}".format(bestparams[1])+"\n"+r'$\tau$2:'+"{0:.1f}".format(bestparams[2]))
#    singlexp = 1/avg_dwells*np.exp(-timearray/avg_dwells)
#    plt.plot(timearray, singlexp, 'orange', label = rf'$\tau$:{avg_dwells:.1f}')
#    plt.xlabel('dwell time (sec)')
#    plt.ylabel('log prob. density')
#    plt.legend(fontsize='x-large')
#  #  plt.savefig(f'{len(exp.files)}files_1_2expfit__compared.png', dpi=200)
