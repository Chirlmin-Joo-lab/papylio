# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:33:17 2020

@author: pimam
"""
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import SAfitting_globalfit3states as SAfitting
    import common_PDF
else:
    from trace_analysis.analysis import SAfitting_globalfit3states as SAfitting
    from trace_analysis.analysis import common_PDF
sns.set(style="ticks")
sns.set_color_codes()

def plot(dwells, name, dist='offtime', trace='red', binsize='auto',
         scale='log', style='dots', color='from_trace', fit_result=None):

    if fit_result is not None:
        tcut = fit_result.tcut
        Tcut = fit_result.Tcut
        Ncut = fit_result.Ncut
        if Ncut > 0:
            dwells = dwells[dwells <= Tcut]
    else:
        tcut = dwells.min()
        Tcut = dwells.max()
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

    print('bin edges ', bin_edges)

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

    fig = plt.figure(f'Histogram {trace} {dist}s {name}', figsize=(4, 3), dpi=200)

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
#        if fit_result.model == '1Exp':
#            tau, error = fit_result.value[0], fit_result.error[0]
#            time, fit = common_PDF.Exp1(tau,
#                                        tcut=tcut, Tcut=Tcut)
#            label = f'\n tau={tau:.1f}'
#
#        elif fit_result.model[0] == '2Exp':
#            time, fit = common_PDF.Exp2(p1, tau1, tau2,
#                                        tcut=tcut, Tcut=Tcut)
#            label = f'\n p1={p1:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

        if fit_result.model == '3Exp' or fit_result.model == '3statekinetic':
            p1 = fit_result.p1
            p2 = fit_result.p2
            tau1 = fit_result.tau1
            tau2 = fit_result.tau2
            tau3 = fit_result.tau3
            
            time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                        tcut=tcut, Tmax=Tcut)
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={tau2:.1f}, tau3={tau3:.1f}'

#        elif fit_result.model[0] == '4Exp':
#            print(f'errors: ', errp1, errp2, errp3, err1, err2, err3, err4)
#            time, fit = common_PDF.Exp4(p1, p2, p3, tau1, tau2, tau3, tau4,
#                                        tcut=tcut, Tcut=Tcut)
#            label = f'\n p1={p1:.2f}, p2={p2:.2f}, p3={p3:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}, tau4={int(tau4)}'

        plt.plot(time, fit, color='k', label=f'{fit_result.model}fit{label}')

    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    plt.tight_layout()
    plt.show()
    return fig


def select_bootstrap(bootresults, bestfit, jset, constructs, Nbins=10, model='None', datasetname='data',
                   method='median_t2_v_t3', percent=100):
    # Getting measures and plotting the parameter values found
    Tcut = bestfit.Tcut[0]

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
        # Constraints on tau's

        lwrbnd = [0.1, 0.1, 35]
        uprbnd = [35, 35, 1.5*Tcut]
#        lwrbnd = [0.1, 0.1, 0.1]
#        uprbnd = [1.5*Tcut, 1.5*Tcut, 1.5*Tcut]

        bootresult1 = bootresults.iloc[:, 0:6]
        Nfits = np.size(bootresult1, 0)
        print('Nfits', Nfits)
        idx = np.arange(Nfits)
        idx_bnd = np.arange(Nfits)

        for j in range(len(constructs)):
            bootresult_j = bootresults.iloc[:, j*6:j*6+6]
            print(bootresult_j)

        # Remove global fits for which a parameter has run into its constraints for any construct
            for i in idx:
                checkp12 = bootresult_j.iloc[i,:2] < 0.0011
                check1 = np.divide(bootresult_j.iloc[i,2:-1], lwrbnd) < 1.1
                check2 = np.divide(uprbnd, bootresult_j.iloc[i,2:-1]) < 1.1
                if np.sum(checkp12) + np.sum(check1) + np.sum(check2) > 0:
                    print(f'Fit {i} : Parameters run into boundary')
                    idx_bnd[i] = -30  # -30 instead of NaN as not possible for integer

        i_nobnd = idx[idx_bnd != -30]
        Nfits_bnd = len(i_nobnd)
        print('Nfits within boundaries', Nfits_bnd)
#        i_nobnd = idx
#        Nfits_bnd = Nfits

        bootresults_jset = bootresults.iloc[:, jset*6:jset*6+6]

        bnd_only = np.zeros((Nfits_bnd, dim+1))
        bnd_only[:,0] = bootresults_jset.iloc[i_nobnd, 0].values
        bnd_only[:,1] = bootresults_jset.iloc[i_nobnd, 1].values
        bnd_only[:,2] = 1 - bnd_only[:,0] - bnd_only[:,1]
        bnd_only[:,3:] = bootresults_jset.iloc[i_nobnd, 2:].values

        # Remove outlier fits based on tau2 and tau3 t-test
#        if method == 'confidence_t2_v_t3':
#           Ztest2 = np.abs(boottau2 - avg_array[4]) / std_array[4]* np.sqrt(Nfits)
#           Ztest3 = np.abs(boottau3 - avg_array[5]) / std_array[5]* np.sqrt(Nfits)
#           Ztest23 = (Ztest2 + Ztest3) / np.sqrt(2) # Stouffer's Z-score method for multivariate analysis
#           print('Z test2 ', Ztest2)
#           print('Z test3 ', Ztest3)
#           print('combined Z test', Ztest23)
#           idx2 = i_nobnd[Ztest23 < 6.3] # first order t-test for confidence 90% two-sided
#           Nfits = len(idx2)

        # Remove outlier fits based on relative distance to median tau2 and tau3 values
        if method == 'median_t2_v_t3':
            boottau2 = bootresults_jset.iloc[i_nobnd, 3].values
            boottau3 = bootresults_jset.iloc[i_nobnd, 4].values
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

        # Remove outlier fits based on BIC
        if method == 'BIC':
            fraction = percent/100
            numfits = int(Nfits_bnd*fraction)
            plt.figure()
            Nbins = 40
            plt.hist(bootresults.BIC, bins=Nbins, label='all fits')
            i_sort = np.argsort(bootresults.BIC[i_nobnd])
            idx2 = i_nobnd[i_sort[:numfits]]
#            plt.hist(bootresults.BIC[idx2], bins=10, label='selected')
            plt.hist(bootresults.BIC[i_nobnd], bins=Nbins, label='within boundaries')
            plt.title(f'histogram of BIC values for bootstrap fits')
            plt.legend()
#            plt.title(f'histogram of BIC values for {len(i_nobnd)} bootstrap fits within boundaries')

        Nfits_selected = len(idx2)
        print('Nfits selected', Nfits_selected)

        selected = np.zeros((Nfits_selected, dim+1))
        selected[:,0] = bootresults_jset.iloc[idx2, 0].values
        selected[:,1] = bootresults_jset.iloc[idx2, 1].values
        selected[:,2] = 1 - selected[:,0] - selected[:,1]
        selected[:,3:] = bootresults_jset.iloc[idx2, 2:].values

        avg_array = np.average(selected[:,:-1], axis=0)
        std_array = np.std(selected[:,:-1], axis=0)

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


def select_Nfits(bootresults, bestfit, jset, constructs, Nbins=10, model='None', datasetname='data',
                   method='median_t2_v_t3', percent=100):
    # Getting measures and plotting the parameter values found
    Tcut = bestfit.Tcut[0]

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
        # Constraints on tau's
        lwrbnd = [0.1, 0.1, 0.1]
        uprbnd = [1.5*Tcut, 1.5*Tcut, 1.5*Tcut]
#        lwrbnd = [0.1, 0.1, 35]
#        uprbnd = [35, 35, 1.5*Tcut]
#        lwrbnd = [0.1, 1.5, 1.5]
#        uprbnd = [1.5, 1.5*Tcut, 1.5*Tcut]

        bootresult1 = bootresults.iloc[:, 0:5]
        Nfits = np.size(bootresult1, 0)
        print('Nfits', Nfits)
        idx = np.arange(Nfits)
        idx_bnd = np.arange(Nfits)

        for j in range(len(constructs)):
            bootresult_j = bootresults.iloc[:, j*5:j*5+5]
            print(bootresult_j)

        # Remove global fits for which a parameter has run into its constraints for any construct
            for i in idx:
                checkp12 = bootresult_j.iloc[i,:2] < 0.0011
                check1 = np.divide(bootresult_j.iloc[i,2:], lwrbnd) < 1.1
                check2 = np.divide(uprbnd, bootresult_j.iloc[i,2:]) < 1.1
                if np.sum(checkp12) + np.sum(check1) + np.sum(check2) > 0:
                    print(f'Fit {i} : Parameters run into boundary')
                    idx_bnd[i] = -30  # -30 instead of NaN as not possible for integer

        i_nobnd = idx[idx_bnd != -30]
        Nfits_bnd = len(i_nobnd)
        print('Nfits within boundaries', Nfits_bnd)
#        i_nobnd = idx
#        Nfits_bnd = Nfits

        bootresults_jset = bootresults.iloc[:, jset*5:jset*5+5]

        bnd_only = np.zeros((Nfits_bnd, dim))
        bnd_only[:,0] = bootresults_jset.iloc[i_nobnd, 0].values
        bnd_only[:,1] = bootresults_jset.iloc[i_nobnd, 1].values
        bnd_only[:,2] = 1 - bnd_only[:,0] - bnd_only[:,1]
        bnd_only[:,3:] = bootresults_jset.iloc[i_nobnd, 2:].values

        # Remove outlier fits based on tau2 and tau3 t-test
#        if method == 'confidence_t2_v_t3':
#           Ztest2 = np.abs(boottau2 - avg_array[4]) / std_array[4]* np.sqrt(Nfits)
#           Ztest3 = np.abs(boottau3 - avg_array[5]) / std_array[5]* np.sqrt(Nfits)
#           Ztest23 = (Ztest2 + Ztest3) / np.sqrt(2) # Stouffer's Z-score method for multivariate analysis
#           print('Z test2 ', Ztest2)
#           print('Z test3 ', Ztest3)
#           print('combined Z test', Ztest23)
#           idx2 = i_nobnd[Ztest23 < 6.3] # first order t-test for confidence 90% two-sided
#           Nfits = len(idx2)

        # Remove outlier fits based on relative distance to median tau2 and tau3 values
        if method == 'median_t2_v_t3':
            boottau2 = bootresults_jset.iloc[i_nobnd, 3].values
            boottau3 = bootresults_jset.iloc[i_nobnd, 4].values
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

        # Remove outlier fits based on BIC
        if method == 'BIC':
            fraction = percent/100
            numfits = int(Nfits_bnd*fraction)
            plt.figure()
            Nbins = 40
            plt.hist(bootresults.BIC, bins=Nbins)
            i_sort = np.argsort(bootresults.BIC[i_nobnd])
            idx2 = i_nobnd[i_sort[:numfits]]
            plt.hist(bootresults.BIC[idx2], bins=10)
            plt.title('histogram of BIC values for Nfits')
            plt.figure()
            plt.hist(bootresults.BIC[i_nobnd], bins=Nbins)
            plt.title(f'histogram of BIC values for {len(i_nobnd)} fits within boundaries')

        Nfits_selected = len(idx2)
        print('Nfits selected', Nfits_selected)

        selected = np.zeros((Nfits_selected, dim))
        selected[:,0] = bootresults_jset.iloc[idx2, 0].values
        selected[:,1] = bootresults_jset.iloc[idx2, 1].values
        selected[:,2] = 1 - selected[:,0] - selected[:,1]
        selected[:,3:] = bootresults_jset.iloc[idx2, 2:].values

        avg_array = np.average(selected[:,:], axis=0)
        std_array = np.std(selected[:,:], axis=0)

        Nfits_results = pd.DataFrame({'p1': selected[:,0],
                                     'p2': selected[:,1],
                                     'p3': selected[:,2],
                                     'tau1': selected[:,3], 
                                     'tau2': selected[:,4],
                                     'tau3': selected[:,5]})
#                                     'Ncut': selected[:,6]})
#                                     'BIC': selected[:,7]})

        boot_stats = pd.DataFrame({'avg': avg_array, 'std': std_array,
                                   'param': param})
    Nfits_results = pd.concat((Nfits_results, boot_stats), axis=1)
    print(Nfits_results)

    for d1 in range(dim-1):
        for d2 in range(d1+1, dim):
            xparam_all = bnd_only[:,d1]
            yparam_all = bnd_only[:,d2]
            xparam = selected[:,d1]
            yparam = selected[:,d2]
            plt.figure()
            plt.plot(xparam_all, yparam_all, '.', color='b', label= f'all fits')# within bounds') 
            plt.plot(xparam, yparam, '.', color='r', label=f'selected fits')
            plt.title(f'{param_nm[d1]} versus {param_nm[d2]} selected:{Nfits_selected} {datasetname}')
            plt.xlabel(f'{param_nm[d1]}')
            plt.ylabel(f'{param_nm[d2]}')
            plt.legend()
#            plt.savefig(f'{datasetname}_{percent}% {model} {param[d1]}_vs_{param[d2]} corrplot_Nfits50_noC_onBIC')
#            plt.close()

    return Nfits_results