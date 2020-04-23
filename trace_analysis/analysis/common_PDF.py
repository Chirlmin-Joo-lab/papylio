# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:29:58 2020

@author: iason
"""

import numpy as np

def Exp1(tau, Tmin=0, Tmax=1000):
    time = np.linspace(Tmin, Tmax, 1000)
    exp = 1/tau*np.exp(-time/tau)

    return time, exp

def Exp2(p1, tau1, tau2, Tmin=0, Tmax=1000):
    time = np.linspace(Tmin, Tmax, 1000)
    exp = p1/tau1*np.exp(-time/tau1)+(1-p1)/tau2*np.exp(-time/tau2)

    return time, exp

def Exp3(p1, p2, tau1, tau2, tau3, Tmin=0, Tmax=1000):
    time = np.linspace(Tmin, Tmax, 1000)
    exp = p1/tau1*np.exp(-time/tau1)+p2/tau2*np.exp(-time/tau2) + \
        + (1-p1-p2)/tau3*np.exp(-time/tau3)

    return time, exp