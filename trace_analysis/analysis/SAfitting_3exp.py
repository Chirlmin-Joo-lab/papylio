# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:52:14 2020

@author: pimam
"""
import numpy as np
import pandas as pd

def P3expcut(dwells, params, constraints, Tcut, Ncut, tcut):
    P1, P2, tau1, tau2, tau3 = Param3exp(params, constraints)
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
    return Pi, Pcut, pcut


def Param3exp(params, constraints):
    Z1, Z2, T1, T2, T3 = params
    zK10, zK20, K10, K20, K30, zK11, zK21, K11, K21, K31 = constraints
#    Z1 = np.exp(z1)/(1 + np.exp(z1))*(zK11 - zK10) + zK10
#    Z2 = np.exp(z2)/(1 + np.exp(z2))*(zK21 - zK20) + zK20
    P1 = np.exp(Z1)/(1 + np.exp(Z1) + np.exp(Z2))
    P2 = np.exp(Z2)/(1 + np.exp(Z1) + np.exp(Z2))
    tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
    tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
    tau3 = np.exp(T3)/(1 + np.exp(T3))*(K31 - K30) + K30
    return P1, P2, tau1, tau2, tau3


#def Param3exp(params, constraints):
#    Z1, Z2, T1, T2, T3 = params
#    ZK10, ZK20, K10, K20, K30, ZK11, ZK21, K11, K21, K31 = constraints
#    print(f'Z1 {Z1}  Z2 {Z2}')
#    checkZ1 = np.sum(ZK10 < Z1)*np.sum(Z1 < ZK11)
#    checkZ2 = np.sum(ZK20 < Z2)*np.sum(Z2 < ZK21)
#    if checkZ1*checkZ2 > 0:
#        P1 = np.exp(Z1)/(1 + np.exp(Z1) + np.exp(Z2))
#        P2 = np.exp(Z2)/(1 + np.exp(Z1) + np.exp(Z2))
#        tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
#        tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
#        tau3 = np.exp(T3)/(1 + np.exp(T3))*(K31 - K30) + K30
#    else:
#        return np.nan, np.nan, np.nan, np.nan, np.nan
#    return P1, P2, tau1, tau2, tau3


def BIC(dwells, k, LLike):
    bic = np.log(dwells.size)*k + 2*LLike
    return bic


def LogLikelihood(tbin, Nmole, params, constraints, model, Tcut, Ncut, tcut):
    Pi, Pcut, pcut = model(tbin, params, constraints, Tcut, Ncut, tcut)
    lognormPi = np.log(Pi/(pcut-Pcut))
    LLike = -np.sum(Nmole*lognormPi)
    if Ncut > 0:
        LLike -= Ncut * np.log(Pcut)
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
                        constraints, Tcut, Ncut, tcut, Tstart=100,
                        Tfinal=0.001, delta=0.05, alpha=0.99):
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
#            x_trial[i] = np.random.uniform(np.max([x[i] - delta1, constraints[i]]),
#                                           np.min([x[i] + delta1, uprbnd[i]]))
        for i in range(0, len(x)):
            x_trial[i] = np.random.uniform(x[i] - delta, x[i] + delta)
        x, xstep = Metropolis(objective_function, model, x, x_trial,
                              constraints, T, tbin, Nmole,
                              Tcut, Ncut, tcut, xstep)
    return x, xstep


def Metropolis(f, model, x, x_trial, constraints, T, tbin, Nmole,
               Tcut, Ncut, tcut, xstep):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(tbin, Nmole, x_trial, constraints, model, Tcut, Ncut, tcut)
    Vold = f(tbin, Nmole, x, constraints, model, Tcut, Ncut, tcut)
    if Vold > 0 and Vnew > 0:
        if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
            x = x_trial
            xstep += 1
    else:
        print('prob LLike')
    return x, xstep


def Best_of_Nfits_sim_anneal(tbin, Nmole, Nfits, model, x_initial,
                             constraints, Tcut, Ncut, tcut):

    # Perform N fits on data using simmulated annealing
    LLike = np.zeros(Nfits)
    for i in range(0, Nfits):
        fitdataZ, xstep = simulated_annealing(tbin=tbin, Nmole=Nmole,
                                             objective_function=LogLikelihood,
                                             model=model, x_initial=x_initial,
                                             constraints=constraints,
                                             Tcut=Tcut, Ncut=Ncut, tcut=tcut)

        LLike[i] = LogLikelihood(tbin, Nmole, fitdataZ, constraints, model,
                                 Tcut, Ncut, tcut)
        if i == 0:
            fitparamZ = [fitdataZ]
            Nsteps = [xstep]
        else:
            fitparamZ = np.concatenate((fitparamZ, [fitdataZ]), axis=0)
            Nsteps = np.concatenate((Nsteps, [xstep]), axis=0)

    ibestparam = np.argmin(LLike)
    bestvaluesZ = fitparamZ[ibestparam]
    bestNsteps = Nsteps[ibestparam]

    return bestvaluesZ, bestNsteps, fitparamZ, LLike


def Bootstrap_data(dwells, Ncut):
    dwells_Ncut = np.concatenate((dwells, np.zeros(Ncut)))
    dwells_rand = np.random.choice(dwells_Ncut, dwells_Ncut.size)
    Bootstrapped_dwells = dwells_rand[dwells_rand > 0]
    Bootstrapped_Ncut = dwells_rand[dwells_rand == 0].size
    return Bootstrapped_dwells, Bootstrapped_Ncut

def fit(dwells_all, mdl, dataset_name='Dwells', Nfits=1, bsize=0, tcut=0,
        Tmax='max', include_over_Tmax=True, bootstrap=False, boot_repeats=0):
    if Tmax == 'max':
        Tmax = dwells_all.max()
    else:
        Tmax = float(Tmax)
    # Calculate Ncut if selected
    if include_over_Tmax:
        Tmax = Tmax - 2 # 2 sec is arbitrary
        dwells = dwells_all[dwells_all < Tmax]
        Ncut = dwells_all[dwells_all >= Tmax].size
        print(f'Ncut: {Ncut}')
    else:
        Ncut = 0
        dwells = dwells_all

    # Bin the data if coarsed-grained fitting is used
    if bsize > 0:
        print('Data coarse-grained for fitting')
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

    if mdl == '3Exp':
        # For 3exp fit the maximum likelihood of the 3exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P3expcut

        # Set parameters for simmulated annealing
#        avg_dwells = np.average(dwells)
        x_initial = [2, 0.1, np.log(0.5), np.log(4.5), np.log(80)]
#        x_initial = np.log([0.8, 0.04, 0.5, 4.5, 80])
#        lwrbnd = [0.001, 0.001, 0.1, 1.2, 40]
#        uprbnd = [1, 1, 1.2, 50, 1.5*Tmax]
        lwrbnd = [-4, -4, 0.1, 0.1, 0.1]
        uprbnd = [4, 4, 1.5*Tmax, 1.5*Tmax, 1.5*Tmax]
        constraints = np.concatenate((lwrbnd, uprbnd))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps, fitparamZ, LLike = Best_of_Nfits_sim_anneal(
                                                           tbin,Nmole, Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tmax, Ncut, tcut)
        
        fitparam = np.full((Nfits, 5), np.nan)
        bic = np.zeros(Nfits)
        for j in range(Nfits):
            params = Param3exp(fitparamZ[j,:], constraints)
            # make sure the fit parameters are ordered from low to high dwelltimes
            imax = np.argmax(params[2:])
            imin = np.argmin(params[2:])
            Parray = [params[0], params[1], 1 - params[0] - params[1]]
            for i in range(0, 3):
                if i != imin and i != imax:
                    imid = i
            params = [Parray[imin], Parray[imid],
                      params[imin+2], params[imid+2],
                      params[imax+2]]
            fitparam[j] = params
            bic[j] = BIC(dwells, 5, LLike[j])

        Nfits_results = pd.DataFrame({'p1': fitparam[:,0],
                                      'p2': fitparam[:,1],
                                      'tau1': fitparam[:,2],
                                      'tau2': fitparam[:,3],
                                      'tau3': fitparam[:,4],
                                      'Ncut': Ncut,
                                      'BIC': bic})
        print(Nfits_results)

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

        # Calculate the BIC for bestvalues
        bestLLike = LogLikelihood(tbin, Nmole, bestvaluesZ, constraints, model,
                                  Tmax, Ncut, tcut)
        bestbic = BIC(dwells, 5, bestLLike)

        # Check whether a parameter has run into its constraints
        check1 = np.divide(bestvalues, lwrbnd) < 1.1 
        check2 = np.divide(uprbnd, bestvalues) < 1.1
        if np.sum(check1) > 0 or np.sum(check2) > 0:
            print('Best param run into boundary')
            print('Best param ', bestvalues)

        errors = [0, 0, 0, 0, 0]
        boot_results = None
        # Check if bootstrapping is used
        if bootstrap:
            lwrbnd = [0.001, 0.001, 0.1, 0.1, 0.1]
            uprbnd = [1, 1, 1.5*Tmax, 1.5*Tmax, 1.5*Tmax]
            constraints = np.concatenate((lwrbnd, uprbnd))

            boot_params = np.full((boot_repeats, 5), np.nan)
            LLike = np.zeros(boot_repeats)
            boot_BIC = np.zeros(boot_repeats)
            Ncutarray = np.full((boot_repeats, 1), np.nan)
            Nstepsarray = np.zeros(boot_repeats)
            print('bootrepeats: ', boot_repeats)
            print('bootparam ', boot_params)
            for j in range(0, boot_repeats):
                boot_dwells, boot_Ncut = Bootstrap_data(dwells, Ncut)
                # Bin the data if coarsed-grained fitting is used
                if bsize > 0:
                    print('Data coarse-grained for fitting')
                    bin_edges = 10**(np.arange(np.log10(min(boot_dwells)),
                                     np.log10(max(boot_dwells)) + bsize, bsize))
                    Nmole, bins = np.histogram(boot_dwells, bins=bin_edges, density=False)
                    tbin = (bins[1:] * bins[:-1])**0.5  # geometric average of bin edges
                    print('N fitbins', len(tbin))
                else:
                    tbin = boot_dwells
                    Nmole = 1
                paramsZ, Nsteps = simulated_annealing(tbin, Nmole,
                                                      LogLikelihood,
                                                      model, x_initial,
                                                      constraints,
                                                      Tcut=Tmax,
                                                      Ncut=boot_Ncut,
                                                      tcut=tcut)
                print(f'boot: {j+1}, steps: {Nsteps}')
                params = Param3exp(paramsZ, constraints)

                # Check whether a parameter has run into its constraints
                check1 = np.divide(params, lwrbnd) < 1.02 
                check2 = np.divide(uprbnd, params) < 1.02
                if np.sum(check1) > 0 or np.sum(check2) > 0:
                    print('Param run into boundary')
                    print('boot params ', params)

                # make sure the fit parameters are ordered from low to high dwelltimes
                imax = np.argmax(params[2:])
                imin = np.argmin(params[2:])
                Parray = [params[0], params[1], 1 - params[0] - params[1]]
                for i in range(0, 3):
                    if i != imin and i != imax:
                        imid = i
                params = [Parray[imin], Parray[imid],
                          params[imin+2], params[imid+2],
                          params[imax+2]]

                Ncutarray[j] = boot_Ncut
                Nstepsarray[j] = Nsteps
                boot_params[j] = params
                LLike[j] = LogLikelihood(tbin, Nmole, paramsZ, constraints,
                                         model, Tmax, boot_Ncut, tcut)
                boot_BIC[j] = BIC(boot_dwells, 5, LLike[j])
            errors = np.std(boot_params, axis=0)
            boot_results = pd.DataFrame({'p1': boot_params[:,0],
                                         'p2': boot_params[:,1],
                                         'tau1': boot_params[:,2],
                                         'tau2': boot_params[:,3],
                                         'tau3': boot_params[:,4],
                                         'Ncut': boot_Ncut,
                                         'BIC': boot_BIC})

        # Put fit result into dataframe
        result = pd.DataFrame({'param': ['p1', 'p2', 'tau1', 'tau2', 'tau3'],
                              'value': bestvalues, 'error': errors})

        result_rest = pd.DataFrame({'Tmax': [Tmax], 'Ncut': [Ncut],
                                    'tcut': [tcut],
                                    'BootRepeats': [boot_repeats*bootstrap],
                                    'steps': [bestNsteps], 'BIC': bestbic})

        fit_result = pd.concat([fit_result, result, result_rest], axis=1)

    return fit_result, boot_results, Nfits_results