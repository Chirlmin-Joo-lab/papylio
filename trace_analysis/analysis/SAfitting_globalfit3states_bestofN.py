# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:52:14 2020

@author: pimam
"""
import numpy as np
import pandas as pd

def P3expcut(dwells, paramsZ, constraints, Tcut, Ncut, tcut):
    P1, P2, tau1, tau2, tau3 = Param3exp(paramsZ, constraints)
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
    return Pi, Pcut, pcut

def P3statekinetic(dwells, paramsZ, constraints, Tcut, Ncut, tcut):
    ku, kB, k12, k21, k23 = Param3state(paramsZ, constraints)
    # Three state model with stuck state which is exited by photobleaching
    a = ku + k12
    b = k21 + k23
    sqroot = np.sqrt((a + b)**2 - 4*(a*b - k21*k12))
    s1= 1/2*(- a - b - sqroot)
    s2= 1/2*(- a - b + sqroot)
    tau1 = -1/s1
    tau2 = -1/s2
    tau3 = 1/kB
    P1 = -(ku*(b+s1)+k12*k23*kB/(s1+kB))/sqroot*tau1
    P2 = (ku*(b+s2)+k12*k23*kB/(s2+kB))/sqroot*tau2

    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
    return Pi, Pcut, pcut


def Param3exp(paramsZ, constraints):
    Z1, Z2, T1, T2, T3 = paramsZ
    zK10, zK20, K10, K20, K30, zK11, zK21, K11, K21, K31 = constraints
    P1 = np.exp(Z1)/(1 + np.exp(Z1) + np.exp(Z2))
    P2 = np.exp(Z2)/(1 + np.exp(Z1) + np.exp(Z2))
    tau1 = np.exp(T1)/(1 + np.exp(T1))*(K11 - K10) + K10
    tau2 = np.exp(T2)/(1 + np.exp(T2))*(K21 - K20) + K20
    tau3 = np.exp(T3)/(1 + np.exp(T3))*(K31 - K30) + K30
    return P1, P2, tau1, tau2, tau3


def Param3state(paramsZ, constraints):
    Z1, Z2, Z3, Z4, Z5 = paramsZ
    LZ1, LZ2, LZ3, LZ4, LZ5, HZ1, HZ2, HZ3, HZ4, HZ5, = constraints
    ku = np.exp(Z1)/(1 + np.exp(Z1))*(HZ1 - LZ1) + LZ1
    kB = np.exp(Z2)/(1 + np.exp(Z2))*(HZ2 - LZ2) + LZ2
    k12 = np.exp(Z3)/(1 + np.exp(Z3))*(HZ3 - LZ3) + LZ3
    k21 = np.exp(Z4)/(1 + np.exp(Z4))*(HZ4 - LZ4) + LZ4
    k23 = np.exp(Z5)/(1 + np.exp(Z5))*(HZ5 - LZ5) + LZ5
    return ku, kB, k12, k21, k23


def BIC(dwells, k, LLike):
    bic = np.log(dwells.size)*k + 2*LLike
    return bic


def LogLikelihood(tbin_ar, Nmole_ar, paramsZ, constraints, model, Tcut, Ncut, tcut):
    LLike = 0
    for ii in range(np.size(tbin_ar, 1)):
        tbin = tbin_ar[:,ii][~np.isnan(tbin_ar[:,ii])]
        Nmole = Nmole_ar[:,ii][~np.isnan(tbin_ar[:,ii])]
        Pi, Pcut, pcut = model(tbin, paramsZ[ii], constraints[ii], Tcut, Ncut[ii], tcut)
        lognormPi = np.log(Pi/(pcut-Pcut))
        LLike -= np.sum(Nmole*lognormPi)
        if Ncut[ii] > 0:
            LLike -= Ncut[ii] * np.log(Pcut)
            print('Ncut used')
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


# SA for all taus constant
#def simulated_annealing(tbin_ar, Nmole_ar, objective_function, model, x_initial,
#                        constraints, Tcut, Ncut, tcut, Tstart=100,
#                        Tfinal=0.001, delta=0.05, alpha=0.99):
#    T = Tstart
#    step = 0
#    xstep = 0
#    x = x_initial
#    while T > Tfinal:
#        step += 1
#        if (step % 100 == 0):
#            T = update_temp(T, alpha)
#        x_trial = np.zeros((np.size(x_initial, 0), np.size(x_initial, 1)))
#        xtau1 = np.random.uniform(x[0][-3] - delta, x[0][-3] + delta)
#        xtau2 = np.random.uniform(x[0][-2] - delta, x[0][-2] + delta)
#        xtau3 = np.random.uniform(x[0][-1] - delta, x[0][-1] + delta)
#        for i in range(0, np.size(x_initial, 0)):
#            for j in range(0, np.size(x_initial, 1)-3):
#                x_trial[i][j] = np.random.uniform(x[i][j] - delta, x[i][j] + delta)
#            x_trial[i][-3] = xtau1
#            x_trial[i][-2] = xtau2
#            x_trial[i][-1] = xtau3
#        x, xstep = Metropolis(objective_function, model, x, x_trial,
#                              constraints, T, tbin_ar, Nmole_ar,
#                              Tcut, Ncut, tcut, xstep)
#    return x, xstep


#SA for kB constant
#def simulated_annealing(tbin_ar, Nmole_ar, objective_function, model, x_initial,
#                        constraints, Tcut, Ncut, tcut, Tstart=100,
#                        Tfinal=0.001, delta=0.05, alpha=0.99):
#    T = Tstart
#    step = 0
#    xstep = 0
#    x = x_initial
#    while T > Tfinal:
#        step += 1
#        if (step % 100 == 0):
#            T = update_temp(T, alpha)
#        x_trial = np.zeros((np.size(x_initial, 0), np.size(x_initial, 1)))
#        xkB = np.random.uniform(x[0][1] - delta, x[0][1] + delta)
#        for i in range(0, np.size(x_initial, 0)):
#            for j in range(0, np.size(x_initial, 1)):
#                x_trial[i][j] = np.random.uniform(x[i][j] - delta, x[i][j] + delta)
#            x_trial[i][1] = xkB
#        x, xstep = Metropolis(objective_function, model, x, x_trial,
#                              constraints, T, tbin_ar, Nmole_ar,
#                              Tcut, Ncut, tcut, xstep)
#    return x, xstep
    

#  SA for tau3 constant
def simulated_annealing(tbin_ar, Nmole_ar, objective_function, model, x_initial,
                        constraints, Tcut, Ncut, tcut, Tstart=100,
                        Tfinal=0.001, delta=0.05, alpha=0.99):
    T = Tstart
    step = 0
    xstep = 0
    x = x_initial
    while T > Tfinal:
        step += 1
        if (step % 100 == 0):
            T = update_temp(T, alpha)
        x_trial = np.zeros((np.size(x_initial, 0), np.size(x_initial, 1)))
        xtau3 = np.random.uniform(x[0][-1] - delta, x[0][-1] + delta)
        for i in range(0, np.size(x_initial, 0)):
            for j in range(0, np.size(x_initial, 1)):
                x_trial[i][j] = np.random.uniform(x[i][j] - delta, x[i][j] + delta)
            x_trial[i][-1] = xtau3
        x, xstep = Metropolis(objective_function, model, x, x_trial,
                              constraints, T, tbin_ar, Nmole_ar,
                              Tcut, Ncut, tcut, xstep)
    return x, xstep


##  SA for tau1 constant
#def simulated_annealing(tbin_ar, Nmole_ar, objective_function, model, x_initial,
#                        constraints, Tcut, Ncut, tcut, Tstart=100,
#                        Tfinal=0.001, delta=0.05, alpha=0.99):
#    T = Tstart
#    step = 0
#    xstep = 0
#    x = x_initial
#    while T > Tfinal:
#        step += 1
#        if (step % 100 == 0):
#            T = update_temp(T, alpha)
#        x_trial = np.zeros((np.size(x_initial, 0), np.size(x_initial, 1)))
#        xtau1 = np.random.uniform(x[0][-3] - delta, x[0][-3] + delta)
#        for i in range(0, np.size(x_initial, 0)):
#            for j in range(0, np.size(x_initial, 1)):
#                x_trial[i][j] = np.random.uniform(x[i][j] - delta, x[i][j] + delta)
#            x_trial[i][-3] = xtau1
#        x, xstep = Metropolis(objective_function, model, x, x_trial,
#                              constraints, T, tbin_ar, Nmole_ar,
#                              Tcut, Ncut, tcut, xstep)
#    return x, xstep


def Metropolis(f, model, x, x_trial, constraints, T, tbin_ar, Nmole_ar,
               Tcut, Ncut, tcut, xstep):
    # Metropolis Algorithm to decide if you accept the trial solution.
    Vnew = f(tbin_ar, Nmole_ar, x_trial, constraints, model, Tcut, Ncut, tcut)
    Vold = f(tbin_ar, Nmole_ar, x, constraints, model, Tcut, Ncut, tcut)
    if Vold > 0 and Vnew > 0:
        if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
            x = x_trial
            xstep += 1
    else:
        print('prob LLike')
    return x, xstep


def Best_of_Nfits_sim_anneal(tbin_ar, Nmole_ar, Nfits, model, x_initial,
                             constraints, Tcut, Ncut, tcut):

    # Perform N fits on data using simmulated annealing
    LLike = np.zeros(Nfits)
    for i in range(0, Nfits):
        fitdataZ, xstep = simulated_annealing(tbin_ar, Nmole_ar,
                                             LogLikelihood,
                                             model, x_initial, constraints,
                                             Tcut, Ncut, tcut)

        LLike[i] = LogLikelihood(tbin_ar, Nmole_ar, fitdataZ, constraints, model,
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


def globfit(datasets, mdl, dataset_name='Dwells', Nfits=1, bsize=0, tcut=0,
        Tmax='max', include_over_Tmax=True, bootstrap=False, boot_repeats=0):

    sets_names = datasets.columns.values
    tbin_ar = np.nan*np.ones((2000, len(sets_names)))
    Nmole_ar = np.nan*np.ones((2000, len(sets_names)))
    Tcut_ar = np.nan*np.ones(len(sets_names))
    Ncut = np.zeros(len(sets_names))
    all_dwells = []
    dwellscombined = []

    for ii in range(len(sets_names)):
        if Tmax == 'max':
            Tcut_ar[ii] = datasets.iloc[:,ii].max()
        else:
            Tcut_ar[ii] = float(Tmax)

        # Calculate Ncut if selected
        if include_over_Tmax:
            Tcut_ar[ii] = Tcut_ar[ii] - 2 # 2 sec is arbitrary
            dwells = datasets.iloc[:,ii][datasets.iloc[:,ii] < Tcut_ar[ii]].values
            Ncut[ii] = datasets.iloc[:,ii][datasets.iloc[:,ii] >= Tcut_ar[ii]].size
            print(f'Ncut: {Ncut}')
        else:
            Ncut[ii] = 0
            dwells = datasets.iloc[:,ii].values

        dwells = dwells[~np.isnan(dwells)]

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
        tbin_ar[:np.size(tbin),ii] = tbin
        Nmole_ar[:np.size(tbin),ii] = Nmole
        all_dwells.append(dwells)
        dwellscombined = np.concatenate((dwellscombined, dwells), axis=0)

    Tcut = Tcut_ar.max()
    print('dwellscombined ', dwellscombined.size)
    print('maxTcut ', Tcut)
    print('size tbin ', np.size(tbin_ar, 1))

    # the initial holder for the fit result irrespective of the fit model
    dataname = pd.DataFrame({'Dataset name': [dataset_name], 'Nfits': [Nfits]})
    data_info = pd.DataFrame({'Datasets': sets_names,
                              'tcut': tcut*np.ones(len(sets_names)),
                              'Tcut': Tcut*np.ones(len(sets_names)),
                              'Ncut': Ncut, 'model': [mdl]*len(sets_names)})
    Nfit_res = pd.DataFrame({})
    boot_results = pd.DataFrame({})

    if mdl == '3Exp':
        # For 3exp fit the maximum likelihood of the 3exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P3expcut

        # Set parameters for simmulated annealing
        x_initial = []
        constraints = []
        for ii in range(len(sets_names)):
            x_initial.append([2, 0.1, np.log(0.5), np.log(4.5), np.log(80)])
#            lwrbnd = [-3.5, -3.5, 0.1, 1.5, 1.5]
#            uprbnd = [3.5, 3.5, 1.5, 1.5*Tcut, 1.5*Tcut]
#            lwrbnd = [-3.5, -3.5, 0.1, 0.1, 0.1]
#            uprbnd = [3.5, 3.5, 1.5*Tcut, 1.5*Tcut, 1.5*Tcut]
            lwrbnd = [-3.5, -3.5, 0.1, 0.1, 35]
            uprbnd = [3.5, 3.5, 35, 35, 1.5*Tcut]
            constraints.append(np.concatenate((lwrbnd, uprbnd)))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps, fitparamZ, LLike = Best_of_Nfits_sim_anneal(
                                                           tbin_ar, Nmole_ar,
                                                           Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tcut, Ncut, tcut)

        for j in range(np.size(fitparamZ, 0)):
            if np.sum(fitparamZ[j] == bestvaluesZ) == np.size(bestvaluesZ):
                jbest = j
        bestLLike = LLike[jbest]

        Bestfit_res = pd.DataFrame(np.nan, index=np.arange(Nfits),
                               columns=['p1', 'p2', 'tau1', 'tau2', 'tau3'])
        for ii in range(len(sets_names)):
            fitparam = np.full((Nfits, 5), np.nan)
            for j in range(Nfits):
                params = Param3exp(fitparamZ[j][ii], constraints[ii])
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

                # Check whether a parameter has run into its constraints
                check1 = np.divide(params, lwrbnd) < 1.1 
                check2 = np.divide(uprbnd, params) < 1.1
                if np.sum(check1) > 0 or np.sum(check2) > 0:
                    print('Param run into boundary')
                    print('params ', params)
                fitparam[j] = params
                if j == jbest:
                    Bestfit_res.loc[ii] = params
            Nfit_res[f'{sets_names[ii]} p1'] = fitparam[:,0]
            Nfit_res[f'{sets_names[ii]} p2'] = fitparam[:,1]
            Nfit_res[f'{sets_names[ii]} tau1'] = fitparam[:,2]
            Nfit_res[f'{sets_names[ii]} tau2'] = fitparam[:,3]
            Nfit_res[f'{sets_names[ii]} tau3'] = fitparam[:,4]
            print('fitparam ', fitparam)

        bic = np.zeros(Nfits)
        for j in range(Nfits):
            bic[j] = BIC(dwellscombined, 17, LLike[j])
        Nfit_res[f'BIC'] =  bic
        bestbic = BIC(dwellscombined, 17, bestLLike)
        print('bestbic ', bestbic)
        Best_rest = pd.DataFrame({'BIC': [bestbic],
                                  'Nsteps': [bestNsteps]})

        # Check if bootstrapping is used
        if bootstrap:
            boot_param = np.full((boot_repeats, len(sets_names), 5), np.nan)
            LLike = np.zeros(boot_repeats)
            boot_BIC = np.zeros(boot_repeats)
            Ncutarray = np.full((boot_repeats, len(sets_names)), np.nan)
            Nstepsarray = np.zeros(boot_repeats)
            print('bootrepeats: ', boot_repeats)

            for j in range(0, boot_repeats):
                tbin_ar = np.nan*np.ones((2000, len(sets_names)))
                Nmole_ar = np.nan*np.ones((2000, len(sets_names)))
                boot_Ncut = np.nan*np.ones(len(sets_names))
                boot_dwellscombined = []
                for ii in range(len(sets_names)):
                    boot_dwells, boot_Ncut[ii] = Bootstrap_data(all_dwells[ii], int(Ncut[ii]))
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
                    tbin_ar[:np.size(tbin),ii] = tbin
                    Nmole_ar[:np.size(tbin),ii] = Nmole
                    boot_dwellscombined = np.concatenate((boot_dwellscombined, boot_dwells), axis=0)
                paramsZ, Nsteps, Ztrials, LLtrials = Best_of_Nfits_sim_anneal(
                                                           tbin_ar, Nmole_ar,
                                                           Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tcut, boot_Ncut, tcut)
                print(f'boot: {j+1}, steps: {Nsteps}')

                for ii in range(len(sets_names)):
                    params = Param3exp(paramsZ[ii], constraints[ii])

                    # Check whether a parameter has run into its constraints
                    check1 = np.divide(params, lwrbnd) < 1.1 
                    check2 = np.divide(uprbnd, params) < 1.1
                    if np.sum(check1) > 0 or np.sum(check2) > 0:
                        print('Param run into boundary')
                        print(f'boot params {sets_names[ii]}', params)

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
                    boot_param[j][ii] = params

                Ncutarray[j] = boot_Ncut
                Nstepsarray[j] = Nsteps
                LLike[j] = LogLikelihood(tbin_ar, Nmole_ar, paramsZ, constraints,
                                         model, Tcut, boot_Ncut, tcut)
                boot_BIC[j] = BIC(boot_dwellscombined, 17, LLike[j])
            for ii in range(len(sets_names)):
                boot_results[f'{sets_names[ii]} p1'] = boot_param[:,ii][:,0]
                boot_results[f'{sets_names[ii]} p2'] = boot_param[:,ii][:,1]
                boot_results[f'{sets_names[ii]} tau1'] = boot_param[:,ii][:,2]
                boot_results[f'{sets_names[ii]} tau2'] = boot_param[:,ii][:,3]
                boot_results[f'{sets_names[ii]} tau3'] = boot_param[:,ii][:,4]
                boot_results[f'{sets_names[ii]} Ncut'] = Ncutarray[:,ii]
            boot_results['BIC'] =  boot_BIC
            boot_results['Nsteps'] = Nstepsarray
        
    if mdl == '3statekinetic':
        # For 3exp fit the maximum likelihood of the 3exp model is obtained
        # with simulated annealing minimization of -log(ML)
        model = P3statekinetic
        parammodel = Param3state
        kinerates = ['ku', 'kB', 'k12', 'k21', 'k23']

        # Set parameters for simmulated annealing
        x_initial = []
        constraints = []
        for ii in range(len(sets_names)):
            x_initial.append([np.log(1.5), np.log(0.02), np.log(0.25), np.log(0.14), np.log(0.1)])
            lwrbnd = [0.001, 0.001, 0.001, 0.001, 0.001]
            uprbnd = [3, 3, 3, 3, 3]
            constraints.append(np.concatenate((lwrbnd, uprbnd)))

        # Perform N fits on data using simmulated annealing and select best
        bestvaluesZ, bestNsteps, fitparamZ, LLike = Best_of_Nfits_sim_anneal(
                                                           tbin_ar, Nmole_ar,
                                                           Nfits,
                                                           model, x_initial,
                                                           constraints,
                                                           Tcut, Ncut, tcut)

        for j in range(np.size(fitparamZ, 0)):
            if np.sum(fitparamZ[j] == bestvaluesZ) == np.size(bestvaluesZ):
                jbest = j
        bestLLike = LLike[jbest]

        Bestfit_res = pd.DataFrame(np.nan, index=np.arange(Nfits),
                               columns=kinerates)
        for ii in range(len(sets_names)):
            fitparam = np.full((Nfits, 5), np.nan)
            for j in range(Nfits):
                params = parammodel(fitparamZ[j][ii], constraints[ii])

                # Check whether a parameter has run into its constraints
                check1 = np.divide(params, lwrbnd) < 1.1 
                check2 = np.divide(uprbnd, params) < 1.1
                if np.sum(check1) > 0 or np.sum(check2) > 0:
                    print('Param run into boundary')
                    print('params ', params)

                fitparam[j] = params
                if j == jbest:
                    Bestfit_res.loc[ii] = params
            Nfit_res[f'{sets_names[ii]} {kinerates[0]}'] = fitparam[:,0]
            Nfit_res[f'{sets_names[ii]} {kinerates[1]}'] = fitparam[:,1]
            Nfit_res[f'{sets_names[ii]} {kinerates[2]}'] = fitparam[:,2]
            Nfit_res[f'{sets_names[ii]} {kinerates[3]}'] = fitparam[:,3]
            Nfit_res[f'{sets_names[ii]} {kinerates[4]}'] = fitparam[:,4]
            print('fitparam ', fitparam)

        bic = np.zeros(Nfits)
        for j in range(Nfits):
            bic[j] = BIC(dwellscombined, 5, LLike[j])
        Nfit_res[f'BIC'] =  bic
        bestbic = BIC(dwellscombined, 5, bestLLike)
        print('bestbic ', bestbic)
        Best_rest = pd.DataFrame({'BIC': [bestbic],
                                  'Nsteps': [bestNsteps]})

        # Check if bootstrapping is used
        if bootstrap:
            boot_param = np.full((boot_repeats, len(sets_names), 5), np.nan)
            LLike = np.zeros(boot_repeats)
            boot_BIC = np.zeros(boot_repeats)
            Ncutarray = np.full((boot_repeats, len(sets_names)), np.nan)
            Nstepsarray = np.zeros(boot_repeats)
            print('bootrepeats: ', boot_repeats)

            for j in range(0, boot_repeats):
                tbin_ar = np.nan*np.ones((2000, len(sets_names)))
                Nmole_ar = np.nan*np.ones((2000, len(sets_names)))
                boot_Ncut = np.nan*np.ones(len(sets_names))
                boot_dwellscombined = []
                for ii in range(len(sets_names)):
                    boot_dwells, boot_Ncut[ii] = Bootstrap_data(all_dwells[ii], int(Ncut[ii]))
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
                    tbin_ar[:np.size(tbin),ii] = tbin
                    Nmole_ar[:np.size(tbin),ii] = Nmole
                    boot_dwellscombined = np.concatenate((boot_dwellscombined, boot_dwells), axis=0)
                paramsZ, Nsteps = simulated_annealing(tbin_ar, Nmole_ar,
                                                      LogLikelihood,
                                                      model, x_initial,
                                                      constraints,
                                                      Tcut=Tcut,
                                                      Ncut=boot_Ncut,
                                                      tcut=tcut)
                print(f'boot: {j+1}, steps: {Nsteps}')

                for ii in range(len(sets_names)):
                    params = Param3exp(paramsZ[ii], constraints[ii])

                    # Check whether a parameter has run into its constraints
                    check1 = np.divide(params, lwrbnd) < 1.1 
                    check2 = np.divide(uprbnd, params) < 1.1
                    if np.sum(check1) > 0 or np.sum(check2) > 0:
                        print('Param run into boundary')
                        print(f'boot params {sets_names[ii]}', params)

                    boot_param[j][ii] = params

                Ncutarray[j] = boot_Ncut
                Nstepsarray[j] = Nsteps
                LLike[j] = LogLikelihood(tbin_ar, Nmole_ar, paramsZ, constraints,
                                         model, Tcut, boot_Ncut, tcut)
                boot_BIC[j] = BIC(boot_dwellscombined, 5, LLike[j])
            for ii in range(len(sets_names)):
                boot_results[f'{sets_names[ii]} {kinerates[0]}'] = boot_param[:,ii][:,0]
                boot_results[f'{sets_names[ii]} {kinerates[1]}'] = boot_param[:,ii][:,1]
                boot_results[f'{sets_names[ii]} {kinerates[2]}'] = boot_param[:,ii][:,2]
                boot_results[f'{sets_names[ii]} {kinerates[3]}'] = boot_param[:,ii][:,3]
                boot_results[f'{sets_names[ii]} {kinerates[4]}'] = boot_param[:,ii][:,4]
                boot_results[f'{sets_names[ii]} Ncut'] = Ncutarray[:,ii]
            boot_results['BIC'] =  boot_BIC
            boot_results['Nsteps'] = Nstepsarray

    Nfits_results = pd.concat([dataname, data_info, Nfit_res], axis=1)

    bestfit_results = pd.concat([dataname, data_info, Bestfit_res, Best_rest], axis=1)

    return bestfit_results, boot_results, Nfits_results