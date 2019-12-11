# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:53:03 2017

@author: ikatechis
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import lmfit
from  scipy import optimize
import scipy.stats as st
import time
import os

plt.close("all")
start_time = time.time()

os.chdir("./2exp")

matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams['figure.dpi'] = 200

clear_contents = True
if clear_contents: # https://stackoverflow.com/questions/185936/delete-folder-contents-in-python
    import os
    folders = ['./plot_fits_2exp']
    for folder in folders:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)             



def data_tmax_gen (data, Tmax):
    data_hold = np.empty((3), int)
    k = 0
    for i in range(data.shape[0]):
        times_M = []
        times_N = []
        times_M = data[i][data[i] < Tmax]
        M = len(times_M) 
#        print (M)
        if M >= 2: # select only those data sets with at least some data points
            times_N = data[i]
            times_M = np.array(times_M)
            row = np.array([times_N, times_M, M])
            data_hold = np.vstack((data_hold, row))
            k += 1
#    print(k)
    if data_hold.shape[0] > 1:     
        data_hold = data_hold[1:]
    return data_hold

def exp2(t, tau1, tau2, a, Tmax=0):
    return a/tau1*np.exp(-t/tau1) + (1-a)/tau2*np.exp(-t/tau2)

def exp2_ML(t, params, Tmax=0):
    tau1, tau2, a = params
    return a/tau1 *np.exp(-t/tau1) + (1-a)/tau2*np.exp(-t/tau2)

def exp2_ML_tmax(t, params, Tmax):
    tau1, tau2, a = params
    q1 = np.exp(-Tmax/tau1)
    q2 = np.exp(-Tmax/tau2)
    renorm = a*(1 - q1) + (1-a)*(1-q2) 
    return (a/tau1 *np.exp(-t/tau1) + (1-a)/tau2*np.exp(-t/tau2))/renorm

def exp2_4param(t, tau1, tau2, A1, A2):
    return A1/tau1*np.exp(-t/tau1) + A2/tau2*np.exp(-t/tau2)

def exp2_tmax(t, params, Tmax):
    tau1, tau2, a = params
    q1 = np.exp(-Tmax/tau1)
    q2 = np.exp(-Tmax/tau2)
    return a/tau1 *np.exp(-t/tau1)/(1-q1) + (1-a)/tau2*np.exp(-t/tau2)/(1-q2)


def hist_gen(gen_data, bin_size=0.5, Tmax=20, error_type=""):
    values, errors, centers, bins = [], [], [], []
    for j in range(gen_data.shape[0]) :
        t_N, t_M, M = gen_data[j,:]
        t = t_N
        binn = np.arange(0, Tmax + bin_size, bin_size)
        val, b = np.histogram(t, bins=binn, density=False)
        N = float(len(t_N))
        # calculate probability for each bin
        p = np.copy(val)/N # do not divide by bin
        if error_type == "binom":
            error = np.sqrt(N*p*(1.0-p))
            error = error/N/bin_size
            errors.append(error)

        # Normalize the histogram for the whole dataset (N datapoints)
#        print(val)
        val = val/N/bin_size
#        print(val)
        values.append(val)
        cent = (b[1:] + b[:-1]) / 2.0
        centers.append(cent)
        bins.append(b)
       
    values = np.array(values)
    if error_type == "train":
        errors = np.std(values, axis=0)
        errors = np.tile(errors, (values.shape[0], 1))
    if error_type == "uniform" or not error_type:
        errors = np.ones(values.shape)
    errors = np.array(errors) 
    centers = np.array(centers)
    bins = np.array(bins)
    return values, centers, bins, errors

def LS_2exp_fit(centers, values, errors, no_params="3"):
    if no_params == "3":
        model = lmfit.Model(exp2)
        model.set_param_hint("a", value=1)
        model.set_param_hint("tmax", vary=False)
    elif no_params == "4":
        model = lmfit.Model(exp2_4param)
        model.set_param_hint("A1", value=1)
        model.set_param_hint("A2", value=1)
    model.set_param_hint("tau1", value=2, max=100)
    model.set_param_hint("tau2", value=15, max=100)
    model.make_params()
    tau1, tau2, a, A1, A2 = [], [], [], [], []

    for i in range(values.shape[0]):
        zero_indx = np.where(values[i] == 0)
        val = values[i][values[i] != 0]
        cent = np.delete(centers[i], zero_indx)
        err = np.delete(errors[i], zero_indx)
        weights = 1./err
        #Fit LS with LM algorithm
        try:
            result = model.fit(val, t=cent, weights=weights, fit_kws={'nan_policy': 'omit'})
            tau1.append(result.values["tau1"])
            tau2.append(result.values["tau2"])
            if no_params == "3":
                a.append(result.values["a"])
            elif no_params == "4":
                A1.append(result.values["A1"])
                A2.append(result.values["A2"])
        except TypeError:
#            print("not enough data to fit! Values={}, rep={}".format(val, i))
            tau1.append(np.nan)
            tau2.append(np.nan)
            if no_params == "3":
                a.append(np.nan)
            elif no_params == "4":
                A1.append(np.nan)
                A2.append(np.nan)
    ar = np.array
    if no_params == "3":
        return {"tau1": ar(tau1), "tau2": ar(tau2), "a": ar(a)}
    elif no_params == "4":            
        return {"tau1": ar(tau1), "tau2": ar(tau2), "A1": ar(A1), "A2":ar(A2)}
    

def LogLike(t, model, *params):
    def plug_in_data(par):
        return np.sum(-np.log(model(t, par, *params)))
    return plug_in_data

def ML_2exp_fit(gen_data, model, tmax=0):
    tau1, tau2, a = [], [], []
    for j in range(gen_data.shape[0]) :
        t_N, t_M, M = gen_data[j,:]
        t = t_M
        ta1, ta2, A = optimize.fmin(LogLike(t, model, tmax),  x0=np.array([1,10,0.5]), disp=False)
        tau1.append(ta1)
        tau2.append(ta2)
        a.append(A)
    return {"tau1": np.array(tau1), "tau2": np.array(tau2), "a": np.array(a)}

tau1_fix = 5
tau2_fix = 20
a_fix = 0.5
Tmax = np.array([20, 40, 60, 100, 200])
N = np.array([100, 1000])
train_rep = 200
bin_size = 0.5
plot_it = True

s = (len(N), len(Tmax), train_rep)
z = np.zeros

LS1 = {"tau1":z(s), "tau2":z(s), "a":z(s)}
LS2 = {"tau1":z(s), "tau2":z(s), "A1":z(s), "A2":z(s)}
ML = {"tau1":z(s), "tau2":z(s), "a":z(s)}
ML_tmax = {"tau1":z(s), "tau2":z(s), "a":z(s)}

#Load the master datafile
file_name = "./data/master_N=10000_rep=10000_tau=1.npy"

for i, n in enumerate(N):
     #pick train_rep random rows from master dataset without replacement
#        np.random.seed(5)
        data = np.load(file_name)
        data = data[np.random.choice(data.shape[0], train_rep, replace=False), :]
        # pick up n random elements from each row without replacement
#        np.random.seed(10)
        data_s = data[:, np.random.choice(data.shape[1], n, replace=False)]

        for j, tmax in enumerate(Tmax):
            plt.close("all")
            # find times < tmax and number of in-layers
            data = data_tmax_gen(data_s, tmax)
            print ("Tmax = {}, N = {}".format(tmax, n))
        #generate histogram
        values, centers, bins, errors = hist_gen(data, bin_size=bin_size, Tmax=tmax)
        
        #fitting LS with 3 params
        ls1 = LS_2exp_fit(centers, values, errors, no_params="3")
        LS1["tau1"][i][j][:] , LS1["tau2"][i][j][:] , LS1["a"] [i][j][:] \
           = ls1["tau1"], ls1["tau2"], ls1["a"]

        #fitting LS with 4 params
        ls2 = LS_2exp_fit(centers, values, errors, no_params="4")
        LS2["tau1"][i][j][:] , LS2["tau2"][i][j][:] , LS2["A1"] [i][j][:], LS2["A2"] [i][j][:] \
           = ls2["tau1"], ls2["tau2"], ls2["A1"], ls2["A2"]
           
        #fit ML assuming simple double exp         
        ml = ML_2exp_fit(data, exp2_ML, tmax)
        ML["tau1"][i][j][:] , ML["tau2"][i][j][:] , ML["a"] [i][j][:] \
           = ml["tau1"], ml["tau2"], ml["a"]
       #fit ML assuming double exp with tmax      
        ml_tmax = ML_2exp_fit(data, exp2_ML_tmax, tmax)  
        ML_tmax["tau1"][i][j][:] , ML_tmax["tau2"][i][j][:] , ML_tmax["a"] [i][j][:] \
           = ml_tmax["tau1"], ml_tmax["tau2"], ml_tmax["a"]
           
        if plot_it:
            save_path = "./plot_fits_2exp/" 
            #plot histogram with fits and save
            opt = [ls1, ml, ml_tmax]
#            opt_avrg = [ls1_avrg, ml_avrg]
#            opt_std = [ls1_std, ml_std]
            names = ["LS1", "ML", "MLtmax"]
            colors = ["y", "darkred", "darkblue"]
            # make figure and plot
            name = "fits_N={}_Tmax={}".format(n, tmax)
            plt.figure(name)
            plt.title(name)
            plt.hist(data[0][0], bins=bins[0], normed=True, alpha=0.3)
            plt.axvline(tmax, color="red", ls="--")
            plt.errorbar(centers[0], values[0], yerr=errors, ls="none",
                         marker=".", color="grey", label="centers")
            for k, val in zip(range(len(opt)), opt):
                av = np.average
                avrg = [av(val["tau1"]), av(val["tau2"]), av(val["a"])]
                std = np.std
                stdv = [std(val["tau1"]), std(val["tau2"]), std(val["a"])]
#                label = names[k]+": $\tau_1 = ${:.2f}$ \pm {:.2f},\tau_2 = {:.2f}"
#                " \pm {:.2f}, a = {:.2f} \pm {:.2f}$".format(avrg[0], stdv[0], avrg[1], stdv[1], avrg[2], stdv[2])
                plt.plot(centers[0], exp2(centers[0], val["tau1"][0], val["tau2"][0], val["a"][0]),
                         color=colors[k], label=names[k]+r": $\tau_1 = ${:.2f}$ \pm {:.2f},\tau_2 = {:.2f}"
                " \pm {:.2f}, a = {:.2f} \pm {:.2f}$".format(avrg[0], stdv[0], avrg[1], stdv[1], avrg[2], stdv[2]))
           
            plt.xlim((0, 100))
            plt.ylim(ymin=0)
            plt.legend()
            plt.savefig(save_path+name+".png")
#            plt.close()