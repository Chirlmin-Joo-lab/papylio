# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:32:58 2019

@author: ikatechis
"""

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import SAfitting
    import common_PDF
else:
    from trace_analysis.analysis import SAfitting_based_on_Z_with_tcut as SAfitting
    from trace_analysis.analysis import common_PDF
# import SAfitting
sns.set(style="ticks")
sns.set_color_codes()


def analyze_combined(dwells_data, dataset_name, dist, configuration):
    # For plotting and fitting of red combined with total dwells
    conf = configuration
    # find the Tmax until which data is selected    
    d = apply_config_to_data(dwells_data, dist, conf)
    figures = []
    fit_data = []
    keys_with_data = []  # keys refer to 'red', 'green', 'total', 'FRET'
    dwells = []
    for key in d.keys():
        if d[key].empty:  # check if the dataframe is empty
            print(f'{dist} dataFrame for {key} is empty')
            continue
        keys_with_data.append(key)
        dwells = np.concatenate((dwells, d[key].loc[:, dist].values))
#    dwells = dwells[dwells < 10]
    print(np.size(dwells), 'dwells selected')
    if conf['FitBool']:
        if conf['tcutBool']:
            tcut = 0.8
            dwells = Short_time_cutoff(dwells, tcut)
            print('tcut:', tcut)
        else:
            tcut = 0
        fit_res = fit(dwells, model=conf['model'],
                      dataset_name=dataset_name,
                      Nfits=int(conf['Nfits']),
                      tcut=tcut,
                      include_over_Tmax=conf['TmaxBool'],
                      bootstrap=conf['BootBool'],
                      boot_repeats=int(conf['BootRepeats']))
        fit_data.append(fit_res)
    else:
        fit_res = None
    print(f'plotting {keys_with_data} {dist}')
    figure = plot(dwells, dataset_name, dist, trace=key,
                  binsize=conf['binsize'], tcut=tcut,
                  scale=conf['scale'], style=conf['PlotType'],
                  fit_result=fit_res)
    figures.append(figure)

    if fit_data != []:
        fit_data = pd.concat(fit_data, axis=1, keys=keys_with_data)
    return dwells, figures, fit_data


def Short_time_cutoff(dwells, tcut):
    short_dwells = np.where(dwells < tcut)[0]
    print('Short dwells cut for plot:', len(short_dwells))
    dwells_cut = dwells[dwells >= tcut]
    return dwells_cut


def fit(dwells, model='1Exp', dataset_name='Dwells', Nfits=1, tcut=0,
        include_over_Tmax=True, bootstrap=True, boot_repeats=100):

    if model == '1Exp+2Exp':
        fit_result = []
        for model in ['1Exp', '2Exp']:
            result, boots = SAfitting.fit(dwells, model, dataset_name, Nfits,
                                          tcut, include_over_Tmax,
                                          bootstrap, boot_repeats)
            fit_result.append(result)
        fit_result = pd.concat(fit_result, axis=1, ignore_index=True)
        return fit_result

    fit_result, boots = SAfitting.fit(dwells, model, dataset_name, Nfits,
                                      tcut, include_over_Tmax,
                                      bootstrap, boot_repeats)
    return fit_result


def plot(dwells, name, dist='offtime', trace='red', binsize='auto', tcut=0,
         scale='log', style='dots', color='from_trace', fit_result=None):
    if tcut > 0:
        dwells = Short_time_cutoff(dwells, tcut)

    if fit_result is not None:
        if fit_result.Ncut[0] > 0:
            Tcut = dwells.max() - 5  # 5 sec is kind of arbitrary here
            dwells = dwells[dwells < Tcut]

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

    print('bin edges', bin_edges)
#    # resolution correction bins
#    if scale == 'Log-Log':
#        t_reso = 0.3
#        ismbin = np.where(np.diff(bin_edges) < t_reso)[0]
#        ismbin = np.concatenate((ismbin, [ismbin[-1] + 1]))  # diff only gives index of first compared
#        print('small bins ', ismbin)
#         
#        j = 0
#        while j < len(ismbin):
#            i = j
#            j += 1
#            while j < len(ismbin) and centers[ismbin[j]] - centers[ismbin[i]] < t_reso:
#                j += 1
#            print('startbin ', centers[ismbin[i]])
#            print('endbin ', centers[ismbin[i]+j-i])
#            print('values ', values[ismbin[i]:(ismbin[i]+j-i)])
##           print('mean value', np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1))
#            values[ismbin[i]:(ismbin[i]+j-i)] = np.sum(values[ismbin[i]:(ismbin[i]+j-i)])/(j-i)

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
            # print('jstart ', izeros[i])
            # print('jend ', izeros[i]+(j-i))
            # print('values ', values[izeros[i]:(izeros[i]+j-i+1)])
            # print('mean value', np.sum(values[izeros[i]:(izeros[i]+j-i+1)])/(j-i+1))
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
    if tcut:
        label = label + f', tcut={tcut:.2f}'
    if style == 'dots':
        plt.plot(centers, values, '.', color=color, label=label)
    if style == 'bars':
        plt.bar(centers, values, color=color, label=label,
                width=(bins[1] - bins[0]))
    if style == 'line':
        plt.plot(centers, values, '-', lw=2, color=color, label=label)

    if fit_result is not None:
        if fit_result.model[0] == '1Exp':
            tau = fit_result.value[0]
            error = fit_result.error[0]
            Ncut = fit_result.Ncut[0]
            print(f'plotting 1Exp fit')
            time, fit = common_PDF.Exp1(tau, tcut=tcut,
                                        Tmax=centers[-1])
            label = f'\n tau={tau:.1f}'
            if error != 0:
                label += f'$\pm$ {error:.1f}'

        elif fit_result.model[0] == '2Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            tau1, err1 = fit_result.value[1], fit_result.error[1]
            tau2, err2 = fit_result.value[2], fit_result.error[2]
            Ncut = fit_result.Ncut[0]
            print(fit_result)
            print(f'errors: ', errp1, err1, err2)
            time, fit = common_PDF.Exp2(p1, tau1, tau2, tcut=tcut,
                                        Tmax=centers[-1], log=True)
            label = f'\n p1={p1:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}'

        elif fit_result.model[0] == '3Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            p2, errp2 = fit_result.value[1], fit_result.error[1]
            tau1, err1 = fit_result.value[2], fit_result.error[2]
            tau2, err2 = fit_result.value[3], fit_result.error[3]
            tau3, err3 = fit_result.value[4], fit_result.error[4]
            Ncut = fit_result.Ncut[0]
            print(fit_result)
            print(f'errors: ', errp1, errp2, err1, err2, err3)
            time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3,
                                        tcut=tcut, Tmax=centers[-1])
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}'

        elif fit_result.model[0] == '4Exp':
            p1, errp1 = fit_result.value[0], fit_result.error[0]
            p2, errp2 = fit_result.value[1], fit_result.error[1]
            p3, errp3 = fit_result.value[2], fit_result.error[2]
            tau1, err1 = fit_result.value[3], fit_result.error[3]
            tau2, err2 = fit_result.value[4], fit_result.error[4]
            tau3, err3 = fit_result.value[5], fit_result.error[5]
            tau4, err4 = fit_result.value[6], fit_result.error[6]
            Ncut = fit_result.Ncut[0]
            print(fit_result)
            print(f'errors: ', errp1, errp2, errp3, err1, err2, err3, err4)
            time, fit = common_PDF.Exp4(p1, p2, p3, tau1, tau2, tau3, tau4,
                                        tcut=tcut, Tmax=centers[-1])
            label = f'\n p1={p1:.2f}, p2={p2:.2f}, p3={p3:.2f}, tau1={tau1:.1f}, tau2={int(tau2)}, tau3={int(tau3)}, tau4={int(tau4)}'

        if fit_result.Ncut[0] > 0:
            label = f', Ncut={int(Ncut)}' + label

        plt.plot(time, fit, color='k', label=f'{fit_result.model[0]}fit{label}')

    if scale in ['Log', 'Log-Log']:
        plt.yscale('log')

    if scale == 'Log-Log':
        plt.xscale('log')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{dist} (s)')
    # plt.locator_params(axis='y', nbins=3)
    plt.tight_layout()
    plt.show()
    return fig


def apply_config_to_data(dwells_data, dist, config):
    t_total = dwells_data['time'][dwells_data['trace'] == 'total']
    t_red = dwells_data['time'][dwells_data['trace'] == 'red']
    print('len t_total', len(t_total))
    print('len t_red', len(t_red))
    d_total = dwells_data[dist][dwells_data['trace'] == 'total']
    d_total = d_total[~np.isnan(d_total)]
    d_red = dwells_data[dist][dwells_data['trace'] == 'red']
    d_red = d_red[~np.isnan(d_red)]
    print('#total', len(d_total))
    print('#red', len(d_red))
    if config['trace']['total'] and config['trace']['red'] and dist == 'offtime':
        t_red = dwells_data['time'][dwells_data['trace'] == 'red']
        t_total = dwells_data['time'][dwells_data['trace'] == 'total']
        print('#t_total', len(t_total))
        print('#t_red', len(t_red))
        print('red dwells overlapping total being removed')
        scan_total = np.arange(0, len(t_total), 2)
        mol_check = np.zeros(len(t_red), dtype=bool)
        for i in scan_total:
            mol = t_total.index[i][0]
            for j in range(len(t_red)):
                mol_check[j] = mol in t_red.index[j]
            redtimes = t_red[mol_check].values
            overlap1 = t_total[i] < redtimes
            overlap2 = redtimes < t_total[i+1]
            idrop = 2*np.unique(np.floor(np.where(overlap1*overlap2)[0]/2))
            data_selected = dwells_data[dist][mol]
            for dd in idrop:
                data_selected[dd] = np.NaN
            dwells_data[dist][mol] = data_selected
#            print('mole ', mol)
#            print('redtimes ', redtimes)
#            totaltimes = [t_total[i], t_total[i+1]]
#            print('totaltimes', totaltimes)
#            print(dwells_data[dist][mol])
    d_red = dwells_data[dist][dwells_data['trace'] == 'red']
    d_total = dwells_data[dist][dwells_data['trace'] == 'total']
    d_red = d_red[~np.isnan(d_red)]
    d_total = d_total[~np.isnan(d_total)]
    print('#total2', len(d_total))
    print('#red2', len(d_red))

    d = dwells_data

    # Select the requested sides
    side_list = ['l'*bool(config['side']['left']),
                 'm'*bool(config['side']['middle']),
                 'r'*bool(config['side']['right'])]

    if dist == 'offtime':
        d = d[d.side.isin(side_list)]
    if dist == 'ontime':
        d = d[d.onside.isin(side_list)]
    # apply min, max conditions
    if config['max'] in ['Max', 'max']:
        d = d[d[dist] > float(config['min'])]
    else:
        d = d[d[dist] > float(config['min'])]
        d = d[d[dist] < float(config['max'])]

    data = {}

    # Collect the dwells of the selected types
    for key in config['trace'].keys():
        if config['trace'][key]:
            data[key] = d[d['trace'] == key]
        else:
            pass

    d_total = d[dist][d['trace'] == 'total']
    d_red = d[dist][d['trace'] == 'red']
    d_total = d_total[~np.isnan(d_total)]
    d_red = d_red[~np.isnan(d_red)]
    print('#red3', len(d_red))
    print('#total3', len(d_total))
#    d_total = data['total'][dist]
    d_red = data['red'][dist]
#    d_total = d_total[~np.isnan(d_total)]
    d_red = d_red[~np.isnan(d_red)]
    print('#red4', len(d_red))
#    print('#total4', len(d_total))

    return data


if __name__ == '__main__':
    filename = 'C:/Users/iason/Desktop/traceanalysis/trace_analysis/traces/'
    filename += 'hel0_dwells_data.xlsx'

    data = pd.read_excel(filename, index_col=[0, 1], dtype={'kon' :np.str})
    print(data.shape)
    config = {'trace': {'red': True, 'green': False, 'total': False, 'FRET': False},
         'side': {'left': True, 'middle': True, 'right': True},
         'min': '0', 'max': 'max',
         'scale': 'Normal',
         'PlotType': 'dots',
         'binsize': 'auto',
         'FitBool': True,
         'TmaxBool': False,
         'BootBool': False,
         'model': '2Exp',
         'Nfits': '1',
         'BootRepeats': '5'}

    result = analyze(data, 'test', 'offtime', config)
