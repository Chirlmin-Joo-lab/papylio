# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:29:58 2020

@author: iason
"""

import numpy as np


def Exp1(tau, tcut=0, Tmax=1000, log=False):
    if log is True:
        time = np.logspace(np.log10(tcut), np.log10(Tmax), 1000)
    else:
        time = np.linspace(tcut, Tmax, 1000)
    exp = 1/tau*np.exp(-time/tau)
    pcut = np.exp(-tcut/tau)
    exp = exp/pcut

    return time, exp


def Exp2(p1, tau1, tau2, tcut=0, Tmax=1000, log=False):
    if log is True:
        time = np.logspace(np.log10(tcut), np.log10(Tmax), 1000)
    else:
        time = np.linspace(tcut, Tmax, 1000)
    exp = p1/tau1*np.exp(-time/tau1)+(1-p1)/tau2*np.exp(-time/tau2)
    pcut = p1*np.exp(-tcut/tau1)+(1-p1)*np.exp(-tcut/tau2)
    exp = exp/pcut

    return time, exp


def Exp3(p1, p2, tau1, tau2, tau3, tcut=0, Tmax=1000, log=False):
    if log is True:
        time = np.logspace(np.log10(tcut), np.log10(Tmax), 1000)
    else:
        time = np.linspace(tcut, Tmax, 1000)
    exp = p1/tau1*np.exp(-time/tau1)+p2/tau2*np.exp(-time/tau2) + \
        + (1-p1-p2)/tau3*np.exp(-time/tau3)
    pcut = p1*np.exp(-tcut/tau1)+p2*np.exp(-tcut/tau2) + \
        (1 - p1 - p2)*np.exp(-tcut/tau3)
    Pcut = p1*np.exp(-Tmax/tau1)+p2*np.exp(-Tmax/tau2) + \
        (1 - p1 - p2)*np.exp(-Tmax/tau3)
    exp = exp/(pcut-Pcut)

    return time, exp


def Exp4(p1, p2, p3, tau1, tau2, tau3, tau4, tcut=0, Tmax=1000, log=False):
    if log is True:
        time = np.logspace(np.log10(tcut), np.log10(Tmax), 1000)
    else:
        time = np.linspace(tcut, Tmax, 1000)
    exp = p1/tau1*np.exp(-time/tau1)+p2/tau2*np.exp(-time/tau2) + \
        + p3/tau3*np.exp(-time/tau3) + \
        + (1-p1-p2-p3)/tau4*np.exp(-time/tau4)
    pcut = p1*np.exp(-tcut/tau1)+p2*np.exp(-tcut/tau2) + \
        + p3*np.exp(-tcut/tau3) + \
        + (1-p1-p2-p3)*np.exp(-tcut/tau4)
    exp = exp/pcut

    return time, exp
