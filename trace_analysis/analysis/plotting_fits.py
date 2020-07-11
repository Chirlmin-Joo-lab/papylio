# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:35:16 2020

@author: pimam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trace_analysis.analysis import common_PDF

def comparative_plot(dwells, name, dist='offtime', trace='red', binsize='auto',
         scale='log', style='dots', color='from_trace', fit_result=None):

    if fit_result is not None:
        tcut = fit_result.tcut[0]
        Tmax = fit_result.Tmax[0]
        Ncut = fit_result.Ncut[0]
        if Ncut > 0:
            dwells = dwells[dwells <= Tmax]
    else:
        Tmax = dwells.max()
        tcut = dwells.min()
        Ncut = 0

    try:
        bsize = float(binsize)
        if scale == 'Log-Log':
            bin_edges = 10**(np.arange(np.log10(min(dwells)), np.log10(max(dwells)) + bsize, bsize))
        else:
            bin_edges = np.arange(min(dwells), max(dwells) + bsize, bsize)
    except ValueError:
        if binsize == 'Auto':
            binsize = 'auto'
        bin_edges = binsize

    values, bins = np.histogram(dwells, bins=bin_edges, density=True)

    # Determine position of bins
    if scale == 'Log-Log':
        centers = (bins[1:] * bins[:-1])**0.5  # geometric average of bin edges
    else:
        centers = (bins[1:] + bins[:-1]) / 2.0

    # combine bins until they contain at least one data point (for y-log plots)
    if scale in ['Log', 'Log-Log']:
        izeros = np.where(values == 0)[0]
        print('izeros', izeros)
        j = 0
        while j < len(izeros):
            i = j
            j += 1
            while j < len(izeros) and izeros[j] - izeros[j-1] == 1:
                j += 1
            values[izeros[i]:(izeros[i]+j-i+1)] = np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1)

    plt.figure(figsize=(4, 3), dpi=200)

    if color == 'from_trace':
        if dist == 'offtime':
            color = 'r'*(trace == 'red') + 'g'*(trace == 'green') + \
                    'b'*(trace == 'FRET') + 'sandybrown'*(trace == 'total')
        if dist == 'ontime':
            color = 'firebrick'*(trace == 'red') + 'olive'*(trace == 'green') + \
                    'darkviolet'*(trace == ' FRET') + 'saddlebrown'*(trace == 'total')

    label = f'{dist} pdf, N={dwells.size}'
    if tcut > 0:
        label = label + f', tcut={tcut:.2f}'
    if Ncut > 0:
        label = label + f', Ncut={int(Ncut)}'

    if style == 'dots':
        plt.plot(centers, values, '.', color=color, label=label)
    if style == 'bars':
        plt.bar(centers, values, color=color, label=label,
                width=(bins[1] - bins[0]))
    if style == 'line':
        plt.plot(centers, values, '-', lw=2, color=color, label=label)

    if fit_result is not None:
        print(fit_result)
        idx_mdl = np.where(~pd.isnull(fit_result.model))[0]
        BICs = np.zeros(3)
        for i in idx_mdl:
            if fit_result.model[i] == '1Exp':
#                BICs[0] = fit_result.BIC[i]
                tau, error = fit_result.value[i], fit_result.error[i]
                time, fit = common_PDF.Exp1(tau,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n tau={tau:.1f}'
                if error != 0:
                    label += f'$\pm$ {error:.1f}'

            if fit_result.model[i] == '2Exp':
                BICs[0] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                tau1, err1 = fit_result.value[i+1], fit_result.error[i+1]
                tau2, err2 = fit_result.value[i+2], fit_result.error[i+2]
                print(f'errors: ', errp1, err1, err2)
                time, fit = common_PDF.Exp2(p1, tau1, tau2,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n p1={p1:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

            if fit_result.model[i] == '3Exp':
                BICs[1] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                p2, errp2 = fit_result.value[i+1], fit_result.error[i+1]
                tau1, err1 = fit_result.value[i+2], fit_result.error[i+2]
                tau2, err2 = fit_result.value[i+3], fit_result.error[i+3]
                tau3, err3 = fit_result.value[i+4], fit_result.error[i+4]
                print(f'errors: ', errp1, errp2, err1, err2, err3)
                time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                            tcut, Tmax)
                label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={tau2:.1f}, tau3={tau3:.1f}'

            if fit_result.model[i] == '4Exp':
                BICs[2] = fit_result.BIC[i]
                p1, errp1 = fit_result.value[i], fit_result.error[i]
                p2, errp2 = fit_result.value[i+1], fit_result.error[i+1]
                p3, errp3 = fit_result.value[i+2], fit_result.error[i+2]
                tau1, err1 = fit_result.value[i+3], fit_result.error[i+3]
                tau2, err2 = fit_result.value[i+4], fit_result.error[i+4]
                tau3, err3 = fit_result.value[i+5], fit_result.error[i+5]
                tau4, err4 = fit_result.value[i+6], fit_result.error[i+6]
                print(f'errors: ', errp1, errp2, errp3, err1, err2, err3, err4)
                time, fit = common_PDF.Exp4(p1, p2, p3, tau1, tau2, tau3, tau4,
                                            tcut=tcut, Tmax=Tmax)
                label = f'\n p1={p1:.2f}, p2={p2:.2f}, p3={p3:.3f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}, tau4={int(tau4)}'

            plt.plot(time, fit, label=f'{fit_result.model[i]}fit{label}')

    numdim = np.arange(len(BICs))
    BICs = BICs - BICs.min()
    print('BICS',BICs)

    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    plt.tight_layout()

    plt.figure('BICplot')
    numdim = [2, 3, 4]#np.arange(2, len(BICs), dtype=int)
    BICs = BICs - BICs.min()
    plt.plot(numdim, BICs, '-r')
    plt.xticks(numdim)
    plt.xlabel('Number of binding phases')
    plt.ylabel('$\Delta$BIC')
    plt.savefig(f'BICplot_offtimes_{trace}_{name}.png', dpi=200)

    plt.show()
    return

def select_bootstrap(bootresults, bestfit, Nbins=10, model='None', datasetname='data',
                   method='confidence_t2_v_t3', percent=100):
     # Getting measures and plotting the parameter values found
     # (up to 4expfit implemented)

    Nfits = np.size(bootresults, 0)
    Tcut = bestfit.Tcut[0]
    print('Nfits', Nfits)

    #Plot correlation plots params to show selection of fits
    if model == '2Exp':
        param_nm = ['p1', r'$\tau$1', r'$\tau$2']
        param = ['p1', 'tau1', 'tau2']
    if model == '3Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3']
    if model == '4Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3', r'$\tau$4']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3', 'tau4']
    if model == '3statekinetic':
        param_nm = ['$k_u$', '$k_B$', '$k_{12}$', '$k_{21}$', '$k_{23}$']
        param = ['ku', 'kB', 'k12', 'k21', 'k23']

    dim = len(param)
    print( '# of params', dim)

    if model == '3Exp':
        #Constraints on tau's
        lwrbnd = [0.1, 0.1, 0.1]
        uprbnd = [1.5*Tcut, 1.5*Tcut, 1.5*Tcut]

        # Remove fits for which a parameter has run into its constraints
        idx = np.arange(Nfits)
        for i in idx:
            checkp12 = bootresults.iloc[i,:2] < 0.0011
            check1 = np.divide(bootresults.iloc[i,2:-2], lwrbnd) < 1.1
            check2 = np.divide(uprbnd, bootresults.iloc[i,2:-2]) < 1.1
            if np.sum(checkp12) + np.sum(check1) + np.sum(check2) > 0:
                print(f'Fit {i} : Parameters run into boundary')
                idx[i] = -30  # -30 instead of NaN as not possible for integer

        i_nobnd = idx[idx != -30]
        Nfits_bnd = len(i_nobnd)
        print('Nfits within boundaries', Nfits_bnd)

        bnd_only = np.zeros((Nfits_bnd, dim+2))
        bnd_only[:,0] = bootresults.iloc[i_nobnd, 0].values
        bnd_only[:,1] = bootresults.iloc[i_nobnd, 1].values
        bnd_only[:,2] = 1 - bnd_only[:,0] - bnd_only[:,1]
        bnd_only[:,3:] = bootresults.iloc[i_nobnd, 2:].values

        # Remove outlier fits based on tau2 and tau3 t-test
#        Ztest2 = np.abs(boottau2 - avg_array[4]) / std_array[4]* np.sqrt(Nfits)
#        Ztest3 = np.abs(boottau3 - avg_array[5]) / std_array[5]* np.sqrt(Nfits)
#        Ztest23 = (Ztest2 + Ztest3) / np.sqrt(2) # Stouffer's Z-score method for multivariate analysis
#        print('Z test2 ', Ztest2)
#        print('Z test3 ', Ztest3)
#        print('combined Z test', Ztest23)
#        idx2 = i_nobnd[Ztest23 < 6.3] # first order t-test for confidence 90% two-sided
#        Nfits = len(idx2)
        
        # Remove outlier fits based on BIC
        if method == 'BIC':
            fraction = percent/100
            numfits = int(Nfits_bnd*fraction)
            plt.figure()
            Nbins = 40
            plt.hist(bootresults.BIC, bins=Nbins, label='within boundaries')
            i_sort = np.argsort(bootresults.BIC[i_nobnd])
            idx2 = i_nobnd[i_sort[:numfits]]
#            plt.hist(bootresults.BIC[idx2], bins=10, label='selected')
            plt.title('histogram of BIC values for bootstrap fits')

        # Remove outlier fits based on relative distance to median tau2 and tau3 values
        if method == 'median_t2_v_t3':
            boottau2 = bootresults.iloc[i_nobnd, 3].values
            boottau3 = bootresults.iloc[i_nobnd, 4].values
            fraction = 1 - percent/100
            num_out = int(Nfits_bnd*fraction)
            if num_out > 0:
                dist1 = (boottau2 - np.median(boottau2)) / np.median(boottau2)
                dist2 = (boottau3 - np.median(boottau3)) / np.median(boottau3)
                dist = np.sqrt(dist1**2 + dist2**2)
                i_sort = np.argsort(dist)
                idx2 = i_nobnd[i_sort[:-num_out]]
            else:
                idx2 = i_nobnd
        
        Nfits_selected = len(idx2)
        print('Nfits selected', Nfits_selected)
        selected = np.zeros((Nfits_selected, dim+2))
        print('Nfits selected', Nfits_selected)
        selected[:,0] = bootresults.iloc[idx2, 0].values
        selected[:,1] = bootresults.iloc[idx2, 1].values
        selected[:,2] = 1 - selected[:,0] - selected[:,1]
        selected[:,3:] = bootresults.iloc[idx2, 2:].values

        avg_array = np.average(selected[:,:-2], axis=0)
        std_array = np.std(selected[:,:-2], axis=0)

        boot_results = pd.DataFrame({'p1': selected[:,0],
                                     'p2': selected[:,1],
                                     'p3': selected[:,2],
                                     'tau1': selected[:,3], 
                                     'tau2': selected[:,4],
                                     'tau3': selected[:,5],
                                     'Ncut': selected[:,6]})
#                                     'BIC': selected[:,7]})

        boot_stats = pd.DataFrame({'avg': avg_array, 'std': std_array,
                                   'param': param})
    boot_results = pd.concat((boot_results, boot_stats), axis=1)
    print(boot_results)

    for d1 in range(dim-1):
        for d2 in range(d1+1, dim):
            xparam_all = bnd_only[:,d1]
            yparam_all = bnd_only[:,d2]
            xparam = selected[:,d1]
            yparam = selected[:,d2]
            plt.figure()
            plt.plot(xparam_all, yparam_all, '.', color='b', label= f'fits within bounds') 
            plt.plot(xparam, yparam, '.', color='r', label=f'selected fits')
            plt.title(f'{param_nm[d1]} versus {param_nm[d2]} selected:{Nfits_selected} {datasetname}')
            plt.xlabel(f'{param_nm[d1]}')
            plt.ylabel(f'{param_nm[d2]}')
            plt.legend()
            plt.savefig(f'{datasetname}_{percent}% {model} {param[d1]}_vs_{param[d2]} corrplot')
            plt.close()

    return boot_results

def histogram_param(params, model, datasetname, Nbins=10, digit=3, save=True):
    # input: pandas DataFrame (Nfits, params)
    # output: histograms, avg and std values
    #         for eeach parameter
    if model == '2Exp':
        param_nm = ['p1', r'$\tau$1', r'$\tau$2']
        param = ['p1', 'tau1', 'tau2']
    if model == '3Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3']
    if model == '4Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3', r'$\tau$4']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3', 'tau4']
    if model == '3statekinetic':
        param_nm = ['$k_u$', '$k_B$', '$k_{12}$', '$k_{21}$', '$k_{23}$']
        param = ['ku', 'kB', 'k12', 'k21', 'k23']

    Nfits = np.size(params, 0)
    dim = len(param)
    print( '# of params', dim)
    avg_array = np.zeros((dim))
    std_array = np.zeros((dim))
    for d in range(dim):
        avg_array[d] = np.average(params.iloc[:,d])
        std_array[d] = np.std(params.iloc[:,d])
        plt.figure()
        plt.hist(params.iloc[:,d], bins=Nbins)
        plt.vlines(avg_array[d], 0, round(Nbins/2), label=f'avg:{avg_array[d]:.{digit}f} std:{std_array[d]:.{digit}f}')
        plt.title(f'Values for {param_nm[d]} Nfits: {Nfits} Nbins: {Nbins}')
        plt.xlabel(f'{param_nm[d]}')
        plt.legend()
        if save is True:
            plt.savefig(f'{datasetname} {model} {param[d]} hist')
            plt.close()

    param_stats = pd.DataFrame({'avg': avg_array, 'std': std_array,
                                'param': param})
    param_results = pd.concat((params, param_stats), axis=1)

    return param_results

def correlation_plot(fits, model, datasetname):
    # input: pandas DataFrame (Nfits, params)
    # output: correlation coefficient and correlation plots
    #         between every parameter
    if model == '2Exp':
        param_nm = ['p1', r'$\tau$1', r'$\tau$2']
        param = ['p1', 'tau1', 'tau2']
    if model == '3Exp':
        param_nm = ['p1', 'p2', 'p3',  r'$\tau$1', r'$\tau$2', r'$\tau$3']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3']
    if model == '4Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3', r'$\tau$4']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3', 'tau4']
    if model == '3statekinetic':
        param_nm = ['$k_u$', '$k_B$', '$k_{12}$', '$k_{21}$', '$k_{23}$']
        param = ['ku', 'kB', 'k12', 'k21', 'k23']

    dim = len(param)
    corr = np.zeros(dim)
    print( '# of params', dim)
    for d1 in range(dim-1):
        for d2 in range(d1+1, dim):
            xparam = fits.iloc[:,d1]
            yparam = fits.iloc[:,d2]
            corrco = np.corrcoef(xparam, yparam)
            corr[d1] = corrco[0,1]
            print(corrco)
            plt.figure()
            plt.plot(xparam, yparam, '.', color='r', label=f'corr coef: {corr[d1]:.2f}')
            plt.title(f'{param_nm[d1]} versus {param_nm[d2]} for {len(xparam)} fits {datasetname}')
            plt.xlabel(f'{param_nm[d1]}')
            plt.ylabel(f'{param_nm[d2]}')
            plt.legend()
            plt.savefig(f'{datasetname} {model} {param[d1]}_vs_{param[d2]} corrplot')
            plt.close()

    return corr

def fits_with_errorbar(results, sets, DNAarray, model, datasetname='data'):
    # To plot best fit , avg and std from bootstrap for different datasets, 3exponential
    if model == '2Exp':
        param_nm = ['p1', 'p2', r'$\tau$1', r'$\tau$2']
        param = ['p1', 'p2', 'tau1', 'tau2']
    if model == '3Exp':
        param_nm = ['p1', 'p2', 'p3', r'$\tau$1', r'$\tau$2', r'$\tau$3']
        param = ['p1', 'p2', 'p3', 'tau1', 'tau2', 'tau3']
    if model == '4Exp':
        param_nm = ['p1', 'p2', 'p3', 'p4', r'$\tau$1', r'$\tau$2', r'$\tau$3', r'$\tau$4']
        param = ['p1', 'p2', 'p3', 'p4', 'tau1', 'tau2', 'tau3', 'tau4']
    if model == '3statekinetic':
        param_nm = ['$k_u$', '$k_B$', '$k_{12}$', '$k_{21}$', '$k_{23}$']
        param = ['ku', 'kB', 'k12', 'k21', 'k23']
    
    # For the last p-value holds:
    # PN = 1- sum_i(to N-1) Pi
    # Pstd = sqrt(sum_i(to N-1) Pi^2)

    Nparam = len(param)
    print( '# of params', Nparam)
          
    bestarray3 = np.zeros((len(sets), len(DNAarray), len(param)))
    avgarray3 = np.zeros((len(sets), len(DNAarray), len(param)))
    stdarray3 = np.zeros((len(sets), len(DNAarray), len(param)))

    for pp in range(len(param)):
        for i in range(len(DNAarray)):
            for j in range(len(sets)):
                bestarray3[j][i][pp] = results[f'best_{sets[j]}'][DNAarray[i]][param[pp]]
                avgarray3[j][i][pp] = results[f'avg_{sets[j]}'][DNAarray[i]][param[pp]]
                stdarray3[j][i][pp] = results[f'std_{sets[j]}'][DNAarray[i]][param[pp]]

        plt.figure()
        plt.title(f'Values for {param_nm[pp]} with errors {datasetname}')
        plt.plot(DNAarray,bestarray3[0][:,pp], 'ob')
        plt.errorbar(DNAarray,avgarray3[0][:,pp], yerr=stdarray3[0][:,pp], color='b', label=f'Individual')#'{sets[0]}')
        plt.plot(DNAarray,bestarray3[1][:,pp], 'or')
        plt.errorbar(DNAarray,avgarray3[1][:,pp], yerr=stdarray3[1][:,pp], color='r', label=r'Global $\tau$3')#f'{sets[1]}')
        plt.plot(DNAarray,bestarray3[2][:,pp], 'o', color='sandybrown')
        plt.errorbar(DNAarray,avgarray3[2][:,pp], yerr=stdarray3[2][:,pp], color='sandybrown', label=r'Global all $\tau$s')#f'{sets[2]}')
        plt.legend()
        plt.savefig(f'{datasetname} {model} {param[pp]} values with errors globalcompared')
        plt.close()

    kuk12best = np.zeros((len(sets), len(DNAarray)))
    k21k23best = np.zeros((len(sets), len(DNAarray)))
    k12k21best = np.zeros((len(sets), len(DNAarray)))
    kuk12avg = np.zeros((len(sets), len(DNAarray)))
    k21k23avg = np.zeros((len(sets), len(DNAarray)))
    k12k21avg = np.zeros((len(sets), len(DNAarray)))
    ku123best = np.zeros((len(sets), len(DNAarray)))
    ku123avg = np.zeros((len(sets), len(DNAarray)))
    for j in range(len(sets)):
        kuk12best[j] = bestarray3[j][:,0] + bestarray3[j][:,2]
        kuk12avg[j] = avgarray3[j][:,0] + avgarray3[j][:,2]
        k12k21best[j] = bestarray3[j][:,2] + bestarray3[j][:,3]
        k12k21avg[j] = avgarray3[j][:,2] + avgarray3[j][:,3]
        k21k23best[j] = bestarray3[j][:,3] + bestarray3[j][:,4]
        k21k23avg[j] = avgarray3[j][:,3] + avgarray3[j][:,4]
        ku123best[j] = kuk12best[j] + k21k23best[j]
        ku123avg[j] = kuk12avg[j] + k21k23avg[j]

    plt.figure()
    plt.title(f'Values for {param_nm[0]}+{param_nm[2]} with errors {datasetname}')
    plt.plot(DNAarray, kuk12best[0], 'ob')
    plt.errorbar(DNAarray, kuk12avg[0], yerr=0, color='b', label=f'Individual')#'{sets[0]}')
    plt.plot(DNAarray, kuk12best[1], 'or')
    plt.errorbar(DNAarray, kuk12avg[1], yerr=0, color='r', label=r'Global $\tau$3')#f'{sets[1]}')
    plt.plot(DNAarray, kuk12best[2], 'o', color='sandybrown')
    plt.errorbar(DNAarray, kuk12avg[2], yerr=0, color='sandybrown', label=r'Global all $\tau$s')#f'{sets[2]}')
    plt.legend()
    plt.savefig(f'{datasetname} {model} {param_nm[0]}+{param_nm[2]} values with errors globalcompared')
    plt.close()
    
    plt.figure()
    plt.title(f'Values for {param_nm[2]}+{param_nm[3]} with errors {datasetname}')
    plt.plot(DNAarray, k12k21best[0], 'ob')
    plt.errorbar(DNAarray, k12k21avg[0], yerr=0, color='b', label=f'Individual')#f'{sets[0]}')
    plt.plot(DNAarray, k12k21best[1], 'or')
    plt.errorbar(DNAarray, k12k21avg[1], yerr=0, color='r', label=r'Global $\tau$3')#f'{sets[1]}')
    plt.plot(DNAarray, k12k21best[2], 'o', color='sandybrown')
    plt.errorbar(DNAarray, k12k21avg[2], yerr=0, color='sandybrown', label=r'Global all $\tau$s')#f'{sets[2]}')
    plt.legend()
    plt.savefig(f'{datasetname} {model} {param_nm[2]}+{param_nm[3]} values with errors globalcompared')
    plt.close()

    plt.figure()
    plt.title(f'Values for {param_nm[3]}+{param_nm[4]} with errors {datasetname}')
    plt.plot(DNAarray, k21k23best[0], 'ob')
    plt.errorbar(DNAarray, k21k23avg[0], yerr=0, color='b', label=f'Individual')#f'{sets[0]}')
    plt.plot(DNAarray, k21k23best[1], 'or')
    plt.errorbar(DNAarray, k21k23avg[1], yerr=0, color='r', label=r'Global $\tau$3')#f'{sets[1]}')
    plt.plot(DNAarray, k21k23best[2], 'o', color='sandybrown')
    plt.errorbar(DNAarray, k21k23avg[2], yerr=0, color='sandybrown', label=r'Global all $\tau$s')#f'{sets[2]}')
    plt.legend()
    plt.savefig(f'{datasetname} {model} {param_nm[3]}+{param_nm[4]} values with errors globalcompared')
    plt.close()

    plt.figure()
    plt.title(f'Values for {param_nm[0]}+{param_nm[2]}+{param_nm[3]}+{param_nm[4]} with errors {datasetname}')
    plt.plot(DNAarray, ku123best[0], 'ob')
    plt.errorbar(DNAarray, ku123avg[0], yerr=0, color='b', label=f'Individual')#'{sets[0]}')
    plt.plot(DNAarray, ku123best[1], 'or')
    plt.errorbar(DNAarray, ku123avg[1], yerr=0, color='r', label=r'Global $\tau$3')#'{sets[1]}')
    plt.plot(DNAarray, ku123best[2], 'o', color='sandybrown')
    plt.errorbar(DNAarray, ku123avg[2], yerr=0, color='sandybrown', label=r'Global all $\tau$s')#f'{sets[2]}')
    plt.legend()
    plt.savefig(f'{datasetname} {model} {param_nm[0]}+{param_nm[2]}+{param_nm[3]}+{param_nm[4]} values with errors globalcompared')
    plt.close()