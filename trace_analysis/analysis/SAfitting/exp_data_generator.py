# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 18:58:52 2017

@author: iason
"""
from __future__ import print_function
import numpy as np
import scipy.stats as st
import time
from pathlib import PureWindowsPath


start_time = time.time()

def exp2(t, tau1, tau2, P1, Tmax=0):
    return P1/tau1 *np.exp(-t/tau1) + (1-P1)/tau2*np.exp(-t/tau2)

def exp1_gen (tau, N):
    return np.random.exponential(tau, N)

def exp2_gen(tau1, tau2, a, N):
    class my_pdf(st.rv_continuous):
        def _pdf(self, t, tau1, tau2, a):
            return exp2(t, tau1, tau2, a)  # Normalized over its range
    dist = my_pdf(a=0, b=np.inf, name='my_pdf')
    times = dist.rvs(tau1, tau2, a, size=N)
    return times

tau_fix = 1
tau1_fix = 10
tau2_fix = 200
a_fix = 0.5
N = np.array([10000])#, 1000]) #, 10000])
train_rep = 1
save = True
# select to generate 1 or 2 exponential
exp_type = "2"


for n in N:
#generate the data
        data = np.empty(n)
        for r in range(0, train_rep):
            print("{}/{}".format(r+1, train_rep))
            if exp_type == "2":
                data = np.vstack((data, exp2_gen(tau1_fix, tau2_fix, a_fix, n)))
            if exp_type == "1":
                data = np.vstack((data, exp1_gen(tau_fix, n)))
        data = data[1:, :]   # discard the zeroth  line
        if save:
            if exp_type == "1":
                np.save("./data/1exp_N={}_rep={}_tau={}".format(n, train_rep, tau_fix),
                        data)
            if exp_type == "2":
                np.save("./data/2exp_N={}_rep={}_tau1={}_tau2={}_a={}".format(n, train_rep, tau1_fix,
                        tau2_fix, a_fix), data)


print("--- %s seconds ---" % (time.time() - start_time))
