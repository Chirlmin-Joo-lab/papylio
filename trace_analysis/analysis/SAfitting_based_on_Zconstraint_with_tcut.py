# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:59:11 2019

@author: pimam
"""
if __name__ == '__main__':
    from trace_analysis.analysis import common_PDF
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


def ML1expcut(dwells, Tcut, Ncut, tcut):
    if Ncut == 0:
        MLtau = np.average(dwells)
    else:
        Nrec = dwells.size
        avg_dwells = np.average(dwells)
        MLtau = avg_dwells + Ncut*Tcut/Nrec
    timearray = np.linspace(0, Tcut, 1000)
    pcut = np.exp(-tcut/MLtau)
    P = 1/MLtau*np.exp(-timearray/MLtau) / pcut
    return P, MLtau


def P1expcut(dwells, params, constraints, Tcut, Ncut, tcut):
    #  Only used to calculate LogLikelihood
    tau = params
    Pi = 1/tau*np.exp(-dwells/tau)
    Pcut = np.exp(-Tcut/tau)
    pcut = np.exp(-tcut/tau)
    return Pi, Pcut, pcut


def P2expcut(dwells, params, constraints, Tcut, Ncut, tcut):
    P1, tau1, tau2 = Param2exp(params, constraints)
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
    pcut = P1*np.exp(-tcut/tau1)+(1-P1)*np.exp(-tcut/tau2)
    return Pi, Pcut, pcut


def P3expcut(dwells, params, constraints, Tcut, Ncut, tcut):
    P1, P2, tau1, tau2, tau3 = Param3exp(params, constraints)
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
    return Pi, Pcut, pcut


def P4expcut(dwells, params, constraints, Tcut, Ncut, tcut):
    P1, P2, P3, tau1, tau2, tau3, tau4 = Param4exp(params, constraints)
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        P3/tau3*np.exp(-dwells/tau3) + \
        (1 - P1 - P2 - P3)/tau4*np.exp(-dwells/tau4)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        P3*np.exp(-Tcut/tau3) + \
        (1 - P1 - P2 - P3)*np.exp(-Tcut/tau4)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        P3*np.exp(-tcut/tau3) + \
        (1 - P1 - P2 - P3)*np.exp(-tcut/tau4)
    return Pi, Pcut, pcut


def Param2exp(params, constraints):
    Z1, T1, T2 = params
    PK10, K10, K20, PK11, K11, K21 = constraints
    P1 = np.exp(Z1)/(1 + np.exp(Z1))*(PK11 - PK10) + PK10
    tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
    tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
    return P1, tau1, tau2


def Param3exp(params, constraints):
    Z1, Z2, T1, T2, T3 = params
    PK10, PK20, K10, K20, K30, PK11, PK21, K11, K21, K31 = constraints
    P1 = np.exp(Z1)/(1 + np.exp(Z1))*(PK11 - PK10) + PK10
    P2 = np.exp(Z2)/(1 + np.exp(Z2))*(1 - P1 - PK20) + PK20
    tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
    tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
    tau3 = np.exp(T3)/(1 + np.exp(T3))*(K31 - K30) + K30
    return P1, P2, tau1, tau2, tau3


def Param4exp(params, constraints):
    Z1, Z2, Z3, T1, T2, T3, T4 = params
    PK10, PK20, PK30, K10, K20, K30, K40, PK11, PK21, PK31, K11, K21, K31, K41 = constraints
    P1 = np.exp(Z1)/(1 + np.exp(Z1))*(PK11 - PK10) + PK10
    P2 = np.exp(Z2)/(1 + np.exp(Z2))*(1 - P1 - PK20) + PK20
    P3 = np.exp(Z2)/(1 + np.exp(Z2))*(1 - P1 - P2 - PK30) + PK30
    tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
    tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
    tau3 = np.exp(T3)/(1 + np.exp(T3))*(K31 - K30) + K30
    tau4 = np.exp(T4)/(1 + np.exp(T4))*(K41 - K40) + K40
    return P1, P2, P3, tau1, tau2, tau3, tau4


def BIC(dwells, k, LLike):
    bic = np.log(dwells.size)*k + 2*LLike
    return bic


def LogLikelihood(tbin, Nmole, params, constraints, model, Tcut, Ncut, tcut):
    Pi, Pcut, pcut = model(tbin, params, constraints, Tcut, Ncut, tcut)
    LLikecut = -Ncut * np.log(Pcut)
    LLike = -np.sum(Nmole*np.log(Pi/(pcut-Pcut))) + LLikecut
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


def simulated_annealing(tbin, Nmole, objective_function, model, x_initial,
                        constraints, Tcut, Ncut, tcut, Tstart=100.,
                        Tfinal=0.001, delta1=0.1, delta2=2.5, alpha=0.995):
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
#            x_trial[i] = np.random.uniform(np.max([x[i] - delta1, lwrbnd[i]]),
#                                           np.min([x[i] + delta1, uprbnd[i]]))
            x_trial[i] = np.random.uniform(x[i] - delta1, x[i] + delta1)
        x, xstep = Metropolis(objective_function, model, x, x_trial,
                              constraints, T, tbin, Nmole,
                              Tcut, Ncut, tcut, xstep)

    return x, xstep


def Metropolis(f, model, x, x_trial, constraints, T, tbin, Nmole,
               Tcut, Ncut, tcut, xstep):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(tbin, Nmole, x_trial, constraints, model, Tcut, Ncut, tcut)
    Vold = f(tbin, Nmole, x, constraints, model, Tcut, Ncut, tcut)
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        x = x_trial
        xstep += 1
    return x, xstep


def Best_of_Nfits_sim_anneal(tbin, Nmole, Nfits, model, x_initial,
                             constraints, Tcut, Ncut, tcut):
    
    # Perform N fits on data using simmulated annealing
    LLike = np.empty(Nfits)
    for i in range(0, Nfits):
        fitdata, xstep = simulated_annealing(tbin=tbin, Nmole=Nmole,
                                             objective_function=LogLikelihood,
                                             model=model, x_initial=x_initial,
                                             constraints=constraints,
                                             Tcut=Tcut, Ncut=Ncut, tcut=tcut)                                            
        if i == 0:
            fitparam = [fitdata]
            Nsteps = [xstep]
        else:
            fitparam = np.concatenate((fitparam, [fitdata]), axis=0)
            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)
        LLike[i] = LogLikelihood(tbin, Nmole, fitparam[i], constraints, model,
                                 Tcut, Ncut, tcut)
    ibestparam = np.argmin(LLike)
    bestparam = fitparam[ibestparam]
    bestNsteps = Nsteps[ibestparam]
    return bestparam, bestNsteps


def Bootstrap_data(dwells, Ncut):
    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
    dwells_rand = np.random.choice(dwells_Ncut, dwells_Ncut.size)
    Bootstrapped_dwells = dwells_rand[dwells_rand > 0]
    Bootstrapped_Ncut = dwells_rand[dwells_rand == 0].size
    return Bootstrapped_dwells, Bootstrapped_Ncut


def fit(dwells_all, mdl, dataset_name='Dwells', Nfits=1, bsize=0, tcut=0,
        include_over_Tmax=True, bootstrap=False, boot_repeats=0):
    # Calculate Ncut if selected
    Tmax = dwells_all.max()
    if include_over_Tmax:
        Tmax = Tmax - 2
        dwells = dwells_all[dwells_all < Tmax]
        Ncut = dwells_all[dwells_all >= Tmax].size
        print(f'Ncut: {Ncut}')
    else:
        Ncut = 0
        dwells = dwells_all

    # Bin the data if coarsed-grained fitting is used
    if bsize > 0:
        bin_edges = 10**(np.arange(np.log10(min(dwells)),
                                   np.log10(max(dwells)) + bsize, bsize))
        Nmole, bins = np.histogram(dwells, bins=bin_edges, density=False)
        tbin = (bins[1:] * bins[:-1])**0.5  # geometric average of bin edges
        print('N fitbins', len(tbin))
    else:
        tbin = dwells
        Nmole = 1

    # the initial holder for the fit result irrespective of the fit model
    fit_result = pd.DataFrame({'Dataset': [dataset_name], 'model': [mdl]})

    if mdl == '1Exp':
        #  The 1exp fit is given by analytic solution, just the average dwelltime
        model = P1expcut
        constraints = [0, 0, 0, 0, 0, 0]
        fit, bestvalue = ML1expcut(dwells, Tmax, Ncut, tcut)
        error = 0
        boot_params = np.empty(boot_repeats)
        if bootstrap is True:
            Ncutarray = np.empty(boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                fit_boot, param = ML1expcut(boot_dwells, Tmax, boot_Ncut, tcut)
                Ncutarray[i] = boot_Ncut
                boot_params[i] = param

            error = np.std(boot_params)

        # Calculate the BIC
        LogLike = LogLikelihood(tbin, Nmole, bestvalue, constraints,
                                model, Tmax, Ncut, tcut)
        bic = BIC(dwells, 1, LogLike)

        result = pd.DataFrame({'param': ['tau'], 'value': [bestvalue],
                               'error': [error], 'Tmax': [Tmax],
                               'Ncut': [Ncut], 'tcut': [tcut],
                               'BootRepeats': [boot_repeats*bootstrap],
                               'steps': ['N/A'], 'BIC': bic})

        fit_result = pd.concat([fit_result, result], axis=1)

    elif mdl == '2Exp':
        # For 2exp fit the maximum likelihood of the 2exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P2expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = np.log([1, 1, 20])
        lwrbnd = [0.001, 0.1, 1]
        uprbnd = [1, 20, 2*Tmax]
        constraints = np.concatenate((lwrbnd, uprbnd))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps = Best_of_Nfits_sim_anneal(tbin, Nmole, Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tcut=Tmax,
                                                           Ncut=Ncut,
                                                           tcut=tcut)

        bestvalues = Param2exp(bestvaluesZ, constraints)

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
                paramsZ, Nsteps = simulated_annealing(boot_dwells,
                                                      LogLikelihood,
                                                      model, x_initial,
                                                      constraints,
                                                      Tcut=Tmax,
                                                      Ncut=boot_Ncut,
                                                      tcut=tcut)
                print(f'boot: {i+1}, steps: {Nsteps}')
                params = Param2exp(paramsZ)
                # make sure the fit parameters are ordered from
                # low to high dwelltimes
                if params[1] > params[2]:
                    params = [1-params[0]] + [params[2], params[1]]

                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                boot_params[i] = params
                LLike[i] = LogLikelihood(tbin, Nmole, paramsZ, constraints,
                                         model, Tmax, Ncut, tcut)
            errors = np.std(boot_params, axis=0)

        # Put fit result into dataframe

        result = pd.DataFrame({'param': ['p', 'tau1', 'tau2'],
                              'value': bestvalues, 'error': errors})

        # Calculate the BIC
        LogLike = LogLikelihood(tbin, Nmole, bestvaluesZ, constraints,
                                model, Tmax, Ncut, tcut)
        bic = BIC(dwells, 3, LogLike)

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                                    'tcut': [tcut],
                                    'BootRepeats': [boot_repeats*bootstrap],
                                    'steps': [bestNsteps], 'BIC': bic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)

    elif mdl == '3Exp':
        # For 3exp fit the maximum likelihood of the 3exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P3expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = np.log([0.83, 0.04, 0.5, 4.5, 80])
        lwrbnd = [0.001, 0.001, 0.1, 0.8, 10]
        uprbnd = [1, np.nan, 1, 20, 2*Tmax]
        constraints = np.concatenate((lwrbnd, uprbnd))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps = Best_of_Nfits_sim_anneal(tbin, Nmole, Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tmax, Ncut, tcut)
        bestvalues = Param3exp(bestvaluesZ, constraints)
        # make sure the fit parameters are ordered from low to high dwelltimes
        imax = np.argmax(bestvalues[2:])
        imin = np.argmin(bestvalues[2:])
        Parray = [bestvalues[0], bestvalues[1],
                  1 - bestvalues[0] - bestvalues[1]]
        for i in range(0, 3):
            if i != imin and i != imax:
                imid = i
        bestvalues = (Parray[imin], Parray[imid],
                      bestvalues[imin+2], bestvalues[imid+2],
                      bestvalues[imax+2])

        errors = [0, 0, 0, 0, 0]
        boot_params = np.empty((boot_repeats, 5))
        # Check if bootstrapping is used
        if bootstrap:
            LLike = np.empty(boot_repeats)
            Ncutarray = np.empty(boot_repeats)
            Nstepsarray = np.empty(boot_repeats)
            print('bootrepeats: ', boot_repeats)
            for i in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                paramsZ, Nsteps = simulated_annealing(boot_dwells,
                                                      LogLikelihood,
                                                      model, x_initial,
                                                      constraints,
                                                      Tcut=Tmax,
                                                      Ncut=boot_Ncut,
                                                      tcut=tcut)
                print(f'boot: {i+1}, steps: {Nsteps}')
                params = Param3exp(paramsZ, constraints)
                # make sure the fit parameters are ordered from low to high dwelltimes
                imax = np.argmax(params[2:])
                imin = np.argmin(params[2:])
                Parray = [params[0], params[1], 1 - params[0] - params[1]]
                for i in range(0, 3):
                    if i != imin and i != imax:
                        imid = i
                params = (Parray[imin], Parray[imid],
                          params[imin+2], params[imid+2],
                          params[imax+2])

                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                boot_params[i] = params
                LLike[i] = LogLikelihood(tbin, Nmole, paramsZ, constraints,
                                         model, Tmax, Ncut, tcut)
            errors = np.std(boot_params, axis=0)

        # Put fit result into dataframe

        result = pd.DataFrame({'param': ['p1', 'p2', 'tau1', 'tau2', 'tau3'],
                              'value': bestvalues, 'error': errors})
        # Calculate the BIC
        LogLike = LogLikelihood(tbin, Nmole, bestvaluesZ, constraints, model,
                                Tmax, Ncut, tcut)
        bic = BIC(dwells, 5, LogLike)

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                                    'tcut': [tcut],
                                    'BootRepeats': [boot_repeats*bootstrap],
                                    'steps': [bestNsteps], 'BIC': bic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)

    elif mdl == '4Exp':
        # For 4exp fit the maximum likelihood of the 4exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P4expcut

        # Set parameters for simmulated annealing
        avg_dwells = np.average(dwells)
        x_initial = np.log([1, 1, 1, 0.5*avg_dwells, avg_dwells,
                            2*avg_dwells, 2*avg_dwells])
        print('x_initial ', x_initial)
        lwrbnd = [0.001, 0.001, 0.001, 0.1, 1, 1, 10]
        uprbnd = [1, 1, 1, 2, 30, 80, 2*Tmax]
        constraints = np.concatenate((lwrbnd, uprbnd))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps = Best_of_Nfits_sim_anneal(tbin, Nmole, Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tmax, Ncut, tcut)
        bestvalues = Param4exp(bestvaluesZ, constraints)

        # make sure the fit parameters are ordered from low to high dwelltimes
#        imax = np.argmax(bestvalues[2:])
#        imin = np.argmin(bestvalues[2:])
#        Parray = [bestvalues[0], bestvalues[1], 1 - bestvalues[0] - bestvalues[1]]
#        for i in range(0, 3):
#            if i != imin and i != imax:
#                imid = i
#        bestvalues = (Parray[imin], Parray[imid],
#                      bestvalues[imin+2], bestvalues[imid+2],
#                      bestvalues[imax+2])

        errors = [0, 0, 0, 0, 0, 0, 0]
        boot_params = np.empty((boot_repeats, 7))
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
                                                      Ncut=boot_Ncut,
                                                      tcut=tcut)
                print(f'boot: {i+1}, steps: {Nsteps}')
                params = Param4exp(paramsZ, constraints)
                # make sure the fit parameters are ordered from low to high dwelltimes
#                imax = np.argmax(params[2:])
#                imin = np.argmin(params[2:])
#                Parray = [params[0], params[1], 1 - params[0] - params[1]]
#                for i in range(0, 3):
#                    if i != imin and i != imax:
#                        imid = i
#                params = (Parray[imin], Parray[imid],
#                          params[imin+2], params[imid+2],
#                          params[imax+2])

                Ncutarray[i] = boot_Ncut
                Nstepsarray[i] = Nsteps
                boot_params[i] = params
                LLike[i] = LogLikelihood(tbin, Nmole, paramsZ, constraints,
                                         model, Tmax, Ncut, tcut)
            errors = np.std(boot_params, axis=0)

        # Put fit result into dataframe

        result = pd.DataFrame({'param': ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3', 'tau4'],
                              'value': bestvalues, 'error': errors})
        # Calculate the BIC
        LogLike = LogLikelihood(tbin, Nmole, bestvaluesZ, constraints,
                                model, Tmax, Ncut, tcut)
        bic = BIC(dwells, 7, LogLike)

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                                    'tcut': [tcut],
                                    'BootRepeats': [boot_repeats*bootstrap],
                                    'steps': [bestNsteps], 'BIC': bic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)

    return fit_result, boot_params


if __name__ == '__main__':    
    
    bsize=0.08
    plt.figure()
    bin_edges = 10**(np.arange(np.log10(min(dataout)), np.log10(max(dataout)) + bsize, bsize))
    values, bins = np.histogram(dataout, bins=bin_edges, density=True)
    centers = (bins[1:] * bins[:-1])**0.5 
    plt.plot(centers, values, '.',color='r')
    p1=0.71
    p2=0.000001
    tau1=2.4
    tau2=10
    tau3=50
    Z1 = np.log(p1/(1-p1-p2))
    Z2= np.log(p2/(1-p1-p2))
    T1 = np.log(tau1)
    T2 = np.log(tau2)
    T3 = np.log(tau3)
    model = P3expcut
    LL = LogLikelihood(dataout, [Z1, Z2, T1, T2, T3], model, 0, 0, tcut=0.9)
    bic = BIC(dataout, 5, LL)
    print(f'LL: {LL} BIC: {bic}')
    time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3, tcut=0.9, Tmax=300)
    plt.loglog(time, fit, label= 'fit1')
    plt.legend()

#    # Import data and prepare for fitting
#    path = 'C:/Users/iason/Desktop/traceanalysis/trace_analysis/traces/'
#    filename = 'hel0_dwells_data.xlsx'
#    dwells_all = pd.read_excel(path+filename).offtime.dropna().values
#
#    # Start fitting
#    mdl = '2Exp'
#
#    include_over_Tmax = False
#    Nfits = 1
#    bootstrap = True
#    boot_repeats = 10
#    result, boot = fit(dwells_all, mdl, 'test', Nfits, include_over_Tmax,
#                  bootstrap, boot_repeats)
#    print(result)
#    plt.hist(boot)
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