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
    if Ncut == 0:
        MLtau = np.average(dwells)
    else:
        Nrec = dwells.size
        avg_dwells = np.average(dwells)
        MLtau = avg_dwells + Ncut*Tcut/Nrec
    timearray = np.linspace(0, Tcut, 1000)
    P = 1/MLtau*np.exp(-timearray/MLtau)
    return P, MLtau

def ML2expcut(dwells, params, Tcut, Ncut):  # not used
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


def Best_of_Nfits_sim_anneal(dwells, Nfits, model, x_initial,
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
    ibestparam = np.argmax(LLike)
    bestparam = fitparam[ibestparam]
    bestNsteps = Nsteps[ibestparam]
    return bestparam, bestNsteps


def Bootstrap_data(dwells, Ncut, Ntrial):
    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
    dwells_rand = np.random.choice(dwells_Ncut, Ntrial)
    Bootstrapped_dwells = dwells_rand[dwells_rand > 0]
    Bootstrapped_Ncut = np.count_nonzero(dwells_rand == 0)
    return Bootstrapped_dwells, Bootstrapped_Ncut


#def Bootstrap_sim_anneal(dwells, repeats, model, x_initial,
#                         lwrbnd, uprbnd, Tcut, Ncut):
#    Ntrial = 1000
#    LLike = np.empty(repeats)
#    Ncutarray = np.empty(repeats)
#    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
#    print(f'dwells size: {dwells_Ncut.size}')
#    for i in range(0, repeats):
#        dwells_rand = np.random.choice(dwells_Ncut, Ntrial)
#        dwells = dwells_rand[dwells_rand > 0]
#        Ncut = np.count_nonzero(dwells_rand == 0)
#        Ncutarray[i] = Ncut
#        fitdata, xstep = simmulated_annealing(data=dwells,
#                                              objective_function=LogLikeLihood,
#                                              model=model, x_initial=x_initial,
#                                              lwrbnd=lwrbnd, uprbnd=uprbnd,
#                                              Tcut=Tcut, Ncut=Ncut)
#        print(f"fit{i} found: {fitdata} Ncut: {Ncut}")
#        if i == 0:
#            fitparam = [fitdata]
#            Nsteps = [xstep]
#        else:
#            fitparam = np.concatenate((fitparam, [fitdata]), axis=0)
#            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)
#        LLike[i] = LogLikeLihood(dwells, fitparam[i], model, Tcut, Ncut)
#    ibestparam = np.argmax(LLike)
#    return fitparam, Nsteps, Ncutarray, ibestparam


def fitting(dwells_all, mdl, Nfits, include_over_Tmax=True,
            bootstrap=False, boot_repeats=0):
    Tmax = dwells_all.max()
    if include_over_Tmax is True:
        Tcut = Tmax - 10
        dwells = dwells_all[dwells_all < Tcut]
        Ncut = dwells_all[dwells_all >= Tcut].size
    else:
        Tcut = 0
        Ncut = 0
        dwells = dwells_all
    print(f'Ncut: {Ncut}')

    if mdl == '1Exp':
        model = ML1expcut
        if Nfits > 1:
            print('Multiple fit not applicable for 1Exp fitting (Nfits>1)')
        if bootstrap is True:
            Ntrial = 1000
            Ncutarray = np.empty(boot_repeats+1)
            Nstepsarray = np.full(boot_repeats+1, np.nan)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut, Ntrial)
                fit, param = model(boot_dwells, Tcut, boot_Ncut)
                Ncutarray[i] = boot_Ncut
                if i == 0:
                    params = [100, param, np.nan]
                    fitparam = [params]
                else:
                    params = [100, param, np.nan]
                    fitparam = np.concatenate((fitparam, [params]), axis=0)

            # Save data of interest to dataframe
            bestfit, bestparam = model(dwells, Tcut, Ncut)
            bestparams = [100, bestparam, np.nan]
            Allfitparam = np.concatenate((fitparam, [bestparams]), axis=0)
            data = pd.DataFrame(Allfitparam)
            print("All fitparam: ", data)
            data.columns = ['P1', 'tau1', 'tau2']
            data['Nsteps'] = Nstepsarray
            data['Ncut'] = Ncutarray
            idx = []
            for i in range(len(fitparam)):
                idx.append('fit' + str(i+1))
            idx.append('Bestfit')
            data.index = idx
        else:
            fit, fitparam = ML1expcut(dwells, Tcut, Ncut)
            # Save data of interest to dataframe
            data = pd.DataFrame({'P1': [100], 'tau1': [fitparam], 'tau2':
                                [np.nan], 'Nsteps': [np.nan], 'Ncut': [Ncut]})

    elif mdl == '2Exp':
        model = P2expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = [0.5, avg_dwells, avg_dwells]
        lwrbnd = [0, 0, 0]
        uprbnd = [1, 2*Tmax, 2*Tmax]

        # Check if bootstrapping is used
        if bootstrap is True:
            Ntrial = 1000
            LLike = np.empty(boot_repeats)
            Ncutarray = np.empty(boot_repeats)
            Nstepsarray = np.full(boot_repeats, np.nan)
            print('bootrepeats: ', boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut, Ntrial)
                param, Nsteps = Best_of_Nfits_sim_anneal(
                                                       boot_dwells, Nfits,
                                                       model=model,
                                                       x_initial=x_initial,
                                                       lwrbnd=lwrbnd,
                                                       uprbnd=uprbnd,
                                                       Tcut=Tcut,
                                                       Ncut=boot_Ncut)
                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                if i == 0:
                    fitparam = [param]
                else:
                    fitparam = np.concatenate((fitparam, [param]), axis=0)
                LLike[i] = LogLikeLihood(dwells, fitparam[i], model, Tcut, Ncut)
            ibestparam = np.argmax(LLike)

            # Save data of interest to dataframe
            bestparam = fitparam[ibestparam]
            bestNsteps = Nstepsarray[ibestparam]
            bestNcut = Ncutarray[ibestparam]
            Allfitparam = np.concatenate((fitparam, [bestparam]), axis=0)
            Nstepsarray = np.concatenate((Nstepsarray, [bestNsteps]), axis=0)
            Ncutarray = np.concatenate((Ncutarray, [bestNcut]), axis=0)
            data = pd.DataFrame(Allfitparam)
            data.columns = ['P1', 'tau1', 'tau2']
            data['Nsteps'] = Nstepsarray
            data['Ncut'] = Ncutarray
            idx = []
            for i in range(len(fitparam)):
                idx.append('fit' + str(i+1))
            idx.append('Bestfit')
            data.index = idx

        else:
            # Perform N fits on data using simmulated annealing and select best
            bestparam, bestNsteps = Best_of_Nfits_sim_anneal(
                                                       dwells, Nfits,
                                                       model=model,
                                                       x_initial=x_initial,
                                                       lwrbnd=lwrbnd,
                                                       uprbnd=uprbnd,
                                                       Tcut=Tcut,
                                                       Ncut=Ncut)
            data = pd.DataFrame({'P1': [bestparam[0]], 'tau1': [bestparam[1]],
                                'tau2': [bestparam[2]], 'Nsteps': [bestNsteps],
                                 'Ncut': [Ncut]})


#        # Quick plot of the dwell time histogram and the corresponding fits
#        plt.figure()
#        values, bins = np.histogram(dwells, bins=40, density=True)
#        centers = (bins[1:] + bins[:-1]) / 2.0
#        plt.plot(centers, values, 'r.', label=f'offtimes N={dwells.size}')
#        timearray = np.linspace(0, Tmax, num=1000)
#        for i in range(0, np.size(fitparam, 0)):
#            fit, Pcut = model(timearray, fitparam[i], Tcut, Ncut)
#            plt.plot(timearray, fit, label='fit'+str(i+1))
#        plt.xlabel('dwell time (sec)')
#        plt.ylabel('prob. density')

    return data


if __name__ == '__main__':

    # Import data and prepare for fitting
    filename = '2exp_N=10000_rep=1_tau1=10_tau2=200_a=0.5'
    dwells_all = np.load('./data/2exp_N=10000_rep=1_tau1=10_tau2=200_a=0.5.npy')
    dwells_all = dwells_all[0]

    # Start fitting
    mdl = '2Exp'
    include_over_Tmax = True
    Nfits = 200
    bootstrap = True
    boot_repeats = 200
    fitdata = fitting(dwells_all, mdl, Nfits, include_over_Tmax, bootstrap, boot_repeats)
    print(fitdata)
    if bootstrap is True:
        fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv', index=False)
    else:
        fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_Nfits{Nfits}.csv', index=False)

#    newdata = pd.read_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv')

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
