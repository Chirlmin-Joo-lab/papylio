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


def P2expcut(dwells, params, Tcut, Ncut):
    P1, tau1, tau2 = Param2exp(params)
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
    return Pi, Pcut


def P3expcut(dwells, params, Tcut, Ncut):
    P1, P2, tau1, tau2, tau3 = Param3exp(params)
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    return Pi, Pcut


def Param2exp(params):
    Z1, T1, T2 = params
    P1 = np.exp(Z1)/(1 + np.exp(Z1))
    tau1 = np.exp(T1)
    tau2 = np.exp(T2)
    return P1, tau1, tau2


def Param3exp(params):
    Z1, Z2, T1, T2, T3 = params
    P1 = np.exp(Z1)/(1 + np.exp(Z1) + np.exp(Z2))
    P2 = np.exp(Z2)/(1 + np.exp(Z1) + np.exp(Z2))
    tau1 = np.exp(T1)
    tau2 = np.exp(T2)
    tau3 = np.exp(T3)
    return P1, P2, tau1, tau2, tau3


def CheckP123(params):
    P1, P2, tau1, tau2, tau3 = params
    if P1 + P2 <= 1:
        stepcheck = True
    else:
        stepcheck = False
    return stepcheck


def BIC(dwells, k, LogLike):
    bic = np.log(dwells.size)*k - 2*LogLike
    return bic


def LogLikelihood(dwells, params, model, Tcut, Ncut):
    Pi, Pcut = model(dwells, params, Tcut, Ncut)
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


def simulated_annealing(data, objective_function, model, x_initial, lwrbnd,
                        uprbnd, Tcut, Ncut, Tstart=100.,
#                        Tfinal=0.001, delta1=0.1, delta2=2.5, alpha=0.90):
                        Tfinal=0.001, delta1=1.5, delta2=2.5, alpha=0.99):
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
#        x_trial[0] = np.random.uniform(np.max([x[0] - delta1, lwrbnd[0]]),
#                                       np.min([x[0] + delta1, uprbnd[0]]))
        for i in range(0, len(x)):
            x_trial[i] = np.random.uniform(np.max([x[i] - delta1, lwrbnd[i]]),
                                           np.min([x[i] + delta1, uprbnd[i]]))
        x, xstep = Metropolis(objective_function, model, x, x_trial, T, data,
                              Tcut, Ncut, xstep)

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
        fitdata, xstep = simulated_annealing(data=dwells,
                                             objective_function=LogLikelihood,
                                             model=model, x_initial=x_initial,
                                             lwrbnd=lwrbnd, uprbnd=uprbnd,
                                             Tcut=Tcut, Ncut=Ncut)
        if i == 0:
            fitparam = [fitdata]
            Nsteps = [xstep]
        else:
            fitparam = np.concatenate((fitparam, [fitdata]), axis=0)
            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)
        LLike[i] = LogLikelihood(dwells, fitparam[i], model, Tcut, Ncut)
        bic = BIC(dwells, len(fitparam[i]), LLike[i])
        fitpar = Param2exp(fitparam[i])
        print(f"fit{i} found: {fitpar} \n LL:{LLike[i]} BIC:{bic}")
    ibestparam = np.argmax(LLike)
    bestparam = fitparam[ibestparam]
    bestNsteps = Nsteps[ibestparam]
    return bestparam, bestNsteps


def Bootstrap_data(dwells, Ncut):
    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
    dwells_rand = np.random.choice(dwells_Ncut, dwells_Ncut.size)
    Bootstrapped_dwells = dwells_rand[dwells_rand > 0]
    Bootstrapped_Ncut = dwells_rand[dwells_rand == 0].size
    return Bootstrapped_dwells, Bootstrapped_Ncut


def fit(dwells_all, mdl, dataset_name='Dwells', Nfits=1,
        include_over_Tmax=True, bootstrap=False, boot_repeats=0):
    Tmax = dwells_all.max()
    if include_over_Tmax:
        Tmax = Tmax - 5
        dwells = dwells_all[dwells_all < Tmax]
        Ncut = dwells_all[dwells_all >= Tmax].size
        print(f'Ncut: {Ncut}')
    else:
        Ncut = 0
        dwells = dwells_all

    # the initial holder for the fit result irrespective of the fit model
    fit_result = pd.DataFrame({'Dataset': [dataset_name], 'model': [mdl]})

    if mdl == '1Exp':
        #  The 1exp fit is given by analytic solution, just the average dwelltime
        fit, bestvalue = ML1expcut(dwells, Tmax, Ncut)
        error = 0
        boot_params = np.empty(boot_repeats)
        if bootstrap is True:
            Ncutarray = np.empty(boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                fit_boot, param = ML1expcut(boot_dwells, Tmax, boot_Ncut)
                Ncutarray[i] = boot_Ncut
                boot_params[i] = param

            error = np.std(boot_params)

        result = pd.DataFrame({'param': ['tau'], 'value': [bestvalue],
                               'error': [error], 'Tmax': [Tmax], 'Ncut': [Ncut],
                               'BootRepeats': [boot_repeats*bootstrap],
                               'steps': ['N/A']})

        fit_result = pd.concat([fit_result, result], axis=1)

    elif mdl == '2Exp':
        # For 2exp fit the maximum likelihood of the 2exp model is obtained with
        # simulated annealing minimization of -log(ML)
        model = P2expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = np.log([1, avg_dwells, avg_dwells])
        lwrbnd = [0, 0, 0]
        uprbnd = np.log([10**5, 3*Tmax, 3*Tmax])

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps = Best_of_Nfits_sim_anneal(
                                                        dwells, Nfits,
                                                        model=model,
                                                        x_initial=x_initial,
                                                        lwrbnd=lwrbnd,
                                                        uprbnd=uprbnd,
                                                        Tcut=Tmax,
                                                        Ncut=Ncut)

        bestvalues = Param2exp(bestvaluesZ)

        # make sure the fit parameters are ordered from low to high dwelltimes
        if bestvalues[1] > bestvalues[2]:
            bestvalues = [1-bestvalues[0]] + [bestvalues[2], bestvalues[1]]

        errors = [0, 0, 0]
        boot_params = np.empty((boot_repeats, 3))
        # Check if bootstrapping is used
        if bootstrap:
            LLike = np.empty(boot_repeats)
            Ncutarray = np.empty(boot_repeats)
            Nstepsarray = np.empty(boot_repeats)
            print('bootrepeats: ', boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                paramsZ, Nsteps = simulated_annealing(
                                                    boot_dwells,
                                                    LogLikelihood,
                                                    model=model,
                                                    x_initial=x_initial,
                                                    lwrbnd=lwrbnd,
                                                    uprbnd=uprbnd,
                                                    Tcut=Tmax,
                                                    Ncut=boot_Ncut)
                print(f'boot: {i+1}, steps: {Nsteps}')
                params = Param2exp(paramsZ)
                # make sure the fit parameters are ordered from low to high dwelltimes
                if params[1] > params[2]:
                    params = [1-params[0]] + [params[2], params[1]]

                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                boot_params[i] = params
                LLike[i] = LogLikelihood(dwells,paramsZ, model, Tmax, Ncut)
            errors = np.std(boot_params, axis=0)

        # Put fit result into dataframe

        result = pd.DataFrame({'param': ['p', 'tau1', 'tau2'],
                              'value': bestvalues, 'error': errors})

        # Calculate the BIC
        LogLike = LogLikelihood(dwells, bestvaluesZ, model, Tmax, Ncut)
        bic = BIC(dwells, len(bestvalues), LogLike)

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                               'BootRepeats': [boot_repeats*bootstrap],
                               'steps': [bestNsteps], 'BIC': bic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)


    elif mdl == '3Exp':
        # For 2exp fit the maximum likelihood of the 2exp model is obtained with
        # simulated annealing minimization of -log(ML)
        model = P3expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = np.log([1, 1, 0.5*avg_dwells, 0.5*avg_dwells, avg_dwells])
        print('x_initial ', x_initial)
        lwrbnd = [0, 0, 0, 0, 0]
        uprbnd = np.log([10**5, 10**5, 3*Tmax, 3*Tmax, 3*Tmax])

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps = Best_of_Nfits_sim_anneal(
                                                        dwells, Nfits,
                                                        model=model,
                                                        x_initial=x_initial,
                                                        lwrbnd=lwrbnd,
                                                        uprbnd=uprbnd,
                                                        Tcut=Tmax,
                                                        Ncut=Ncut)
        bestvalues = Param3exp(bestvaluesZ)
        
        # make sure the fit parameters are ordered from low to high dwelltimes
        # if bestvalues[1] > bestvalues[2]:
        #     bestvalues = [1-bestvalues[0]] + [bestvalues[2], bestvalues[1]]

        errors = [0, 0, 0, 0, 0]
        boot_params = np.empty((boot_repeats,5))
        # Check if bootstrapping is used
        if bootstrap:
            LLike = np.empty(boot_repeats)
            Ncutarray = np.empty(boot_repeats)
            Nstepsarray = np.empty(boot_repeats)
            print('bootrepeats: ', boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                paramsZ, Nsteps = simulated_annealing(
                                                      boot_dwells,
                                                      LogLikelihood,
                                                      model=model,
                                                      x_initial=x_initial,
                                                      lwrbnd=lwrbnd,
                                                      uprbnd=uprbnd,
                                                      Tcut=Tmax,
                                                      Ncut=boot_Ncut)
                print(f'boot: {i+1}, steps: {Nsteps}')
                params = Param3exp(paramsZ)
                # make sure the fit parameters are ordered from low to high dwelltimes
                # if params[1] > params[2]:
                #     params = [1-params[0]] + [params[2], params[1]]

                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                boot_params[i] = params
                LLike[i] = LogLikelihood(dwells, paramsZ, model, Tmax, Ncut)
            errors = np.std(boot_params, axis=0)

        # Put fit result into dataframe

        result = pd.DataFrame({'param': ['p1', 'p2', 'tau1', 'tau2', 'tau3'],
                              'value': bestvalues, 'error': errors})
        # Calculate the BIC
        LogLike = LogLikelihood(dwells, bestvaluesZ, model, Tmax, Ncut)
        bic = BIC(dwells, len(bestvalues), LogLike)

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                                    'BootRepeats': [boot_repeats*bootstrap],
                                    'steps': [bestNsteps], 'BIC': bic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)

    return fit_result, boot_params


if __name__ == '__main__':

    # Import data and prepare for fitting
    path = 'C:/Users/iason/Desktop/traceanalysis/trace_analysis/traces/'
    filename = 'hel0_dwells_data.xlsx'
    dwells_all = pd.read_excel(path+filename).offtime.dropna().values

    # Start fitting
    mdl = '2Exp'

    include_over_Tmax = False
    Nfits = 1
    bootstrap = True
    boot_repeats = 10
    result, boot = fit(dwells_all, mdl, 'test', Nfits, include_over_Tmax,
                  bootstrap, boot_repeats)
    print(result)
    plt.hist(boot)
    # if bootstrap is True:
    #     fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv', index=False)
    # else:
    #     fitdata.to_csv(f'{mdl}_inclTmax_{include_over_Tmax}_Nfits{Nfits}.csv', index=False)

#    newdata = pd.read_csv(f'{mdl}_inclTmax_{include_over_Tmax}_bootstrap{boot_repeats}.csv')

    # Getting measures and plotting the parameter values found
    # taubnd = 100
    # fitP1 = []
    # fittau1 = []
    # fittau2 = []
    # for i in range(0, len(fitdata['tau1'])):
    #     if fitdata['tau1'][i] > taubnd:
    #         fittau2.append(fitdata['tau1'][i])
    #         fitP1.append(1-fitdata['P1'][i])
    #     else:
    #         fittau1.append(fitdata['tau1'][i])
    #         fitP1.append(fitdata['P1'][i])
    #     if fitdata['tau2'][i] > taubnd:
    #         fittau2.append(fitdata['tau2'][i])
    #     else:
    #         fittau1.append(fitdata['tau2'][i])

    # P1_avg = np.average(fitP1)
    # tau1_avg = np.average(fittau1)
    # tau2_avg = np.average(fittau2)
    # P1_std = np.std(fitP1)
    # tau1_std = np.std(fittau1)
    # tau2_std = np.std(fittau2)
    # Nbins = 50

    # plt.figure()
    # plt.hist(fitP1, bins=Nbins)
    # plt.vlines(P1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(P1_avg))
    # plt.title(f'Fit values for P1 Nfits: {boot_repeats} Nbins: {Nbins}')
    # plt.legend()
    # plt.figure()
    # plt.hist(fittau1, bins=Nbins)
    # plt.vlines(tau1_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau1_avg))
    # plt.title(rf'Fit values for $\tau$1 Nfits: {boot_repeats} Nbins: {Nbins}')
    # plt.legend()
    # plt.figure()
    # plt.hist(fittau2, bins=Nbins)
    # plt.vlines(tau2_avg, 0, round(Nbins/2), label='avg:'+"{0:.2f}".format(tau2_avg))
    # plt.title(rf'Fit values for $\tau$1 Nfits: {boot_repeats} Nbins: {Nbins}')
    # plt.legend()


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


# def ML2expcut(dwells, params, Tcut, Ncut):  # not used
#     P1, tau1, tau2 = params
#     Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
#     LLike = np.sum(-np.log(Pi))
#     if Ncut != 0:
#         Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
#         LLikecut = -Ncut * np.log(Pcut)
#         LLike += LLikecut
#     return Pi, LLike