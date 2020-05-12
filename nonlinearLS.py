# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:49:44 2020

@author: pimam
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def PDFExp3(params, tcut=0, Tmax=1000, log=False, bsize=0.001):
    if log is True:
        time = 10**(np.arange(np.log10(tcut), np.log10(Tmax) + bsize, bsize))
    else:
        time = np.linspace(tcut, Tmax, 1000)
    p1, p2, tau1, tau2, tau3 = params
    exp = p1/tau1*np.exp(-time/tau1) + p2/tau2*np.exp(-time/tau2) + \
        + (1-p1-p2)/tau3*np.exp(-time/tau3)
    pcut = p1*np.exp(-tcut/tau1) + p2*np.exp(-tcut/tau2) + \
        (1 - p1 - p2)*np.exp(-tcut/tau3)
    Pcut = p1*np.exp(-Tmax/tau1) + p2*np.exp(-Tmax/tau2) + \
        (1 - p1 - p2)*np.exp(-Tmax/tau3)
    exp = exp/(pcut-Pcut)

    return time, exp


def P3expcut(dwells, P1, P2, tau1, tau2, tau3):  # , Tcut, Ncut, tcut):
    tcut = 0.9
    Tcut = dwells.max()
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
    P = np.prod(Pi)/(pcut-Pcut)    
    return P

bsize = 0.04
tcut = 0.9
Tmax = dataout.max()-100
dataout = dataout[dataout< Tmax]
time = 10**(np.arange(np.log10(tcut), np.log10(Tmax) + bsize, bsize))
bsize =0.04
bin_edges = 10**(np.arange(np.log10(min(dataout)), np.log10(max(dataout)) + bsize, bsize))
bins = bin_edges
values, bins = np.histogram(dataout, bins=bin_edges, density=True)
centers = (bins[1:] * bins[:-1])**0.5
plt.plot(centers, values, '.', color='r')

popt, pcov = curve_fit(P3expcut, centers, values, bounds=(0, [1, 1, 1, 15, 100]))
print(popt)
time, exp3fit = PDFExp3(popt, tcut=tcut, Tmax=Tmax, log=True, bsize=bsize)
plt.loglog(time, exp3fit)
