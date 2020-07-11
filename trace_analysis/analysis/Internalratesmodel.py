# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:38:24 2020

@author: pimam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import SAfitting
    import common_PDF
else:
    from trace_analysis.analysis import SAfitting
    from trace_analysis.analysis import common_PDF


def Internal_rates_2state_model(P1, tau1, tau2):
#A double exponential binding time distribution can be described by a 3 state model
    k21 = 1/(tau1*(1-P1) + tau2*P1)
    k13 = P1/tau1 + (1-P1)/tau2
    k12 = (1-P1)/tau1 + P1/tau2 - k21

    return k21, k13, k12

def Internal_rates_3state_model(P1, P2, tau1, tau2, tau3):
# Three state model with stuck state which is exited by photobleaching
    kB = 1/tau3
    ku = P1/tau1 + P2/tau2 + (1-P1-P2)/tau3
    k12 = 1/ku*(1-P1-P2)*(kB-1/tau1)*(kB-1/tau2) + 1/tau1 + 1/tau2 - ku - 1/ku/tau1/tau2
    k12 = 1/ku*((1-P1-P2)*(kB-1/tau1)*(kB-1/tau2) - (ku-1/tau1)*(ku-1/tau2))
    k23 = (1-P1-P2)*(kB-1/tau1)*(kB-1/tau2)/k12
    k21 = 1/tau1 + 1/tau2 - ku - k12 - k23

    return ku, kB, k12, k21, k23

def ThreeExp_from_rates_3state(ku, kB, k12, k21, k23):
# Three state model with stuck state which is exited by photobleaching
    a = ku + k12
    b = k21 + k23
    sqroot = np.sqrt((a + b)**2 - 4*(a*b - k21*k12))
    s1= 1/2*(- a - b - sqroot)
    s2= 1/2*(- a - b + sqroot)
    tau1 = -1/s1
    tau2 = -1/s2
    tau3 = 1/kB
    p1 = -(ku*(b+s1)+k12*k23*kB/(s1+kB))/sqroot*tau1
    p2 = (ku*(b+s2)+k12*k23*kB/(s2+kB))/sqroot*tau2

    return p1, p2, tau1, tau2, tau3

if __name__ == '__main__':   
    p1 = 0.8
    p2 = 0.05
    p3 = 1 - p1 - p2
    tau1 = 0.5
    tau2 = 5
    tau3 = 70

    plt.figure()

    ku, kB, k12, k21, k23 = Internal_rates_3state_model(p1, p2, tau1, tau2, tau3)
    print(f'ku {ku} kB {kB} k12 {k12} k21 {k21} k23 {k23}')

    p1c, p2c, tau1c, tau2c, tau3c = ThreeExp_from_rates_3state(ku, kB, k12, k21, k23)
    p3c = 1 - p1c - p2c
    print(f'tau1 {tau1c} tau2 {tau2c} tau3 {tau3c} P1 {p1c} P2 {p2c} P3 {p3c}')

#    # Calculation of the probability distribution based on the rates of 3state model
#    a = ku + k12
#    b = k21 + k23
#    sqroot = np.sqrt((a + b)**2 - 4*(k12*k23 + b*ku))
#    P1uB = np.exp(-(1/2)*(a + b + sqroot)*t)*ku*(a - b + sqroot)/(2*sqroot) +\
#           np.exp(1/2*(- a - b + sqroot)*t)*(ku*(-a + b + sqroot)/(2*sqroot) -\
#                  k12*k23*kB*(a + b - 2*kB + sqroot) /\
#                  (2*(k12*(k23 - kB) - (b - kB)*(kB - ku))*sqroot)) +\
#           k12*k23*kB*np.exp(-kB*t)/(k12*(k23 - kB) - (b - kB)*(kB - ku))
#
#    plt.plot(t, P1uB, color='r', label=f'kinetic rates model')
#
#    # Back calculation of parameters 3 exponential
#    tau1c= 1/(1/2*(a + b + sqroot))
#    tau2c = -1/(1/2*(- a - b + sqroot))
#    tau3c = 1/kB
#    p1c = ku*(a - b + sqroot)/(2*sqroot)*tau1
#    p2c = (ku*(-a + b + sqroot)/(2*sqroot) - k12*k23*kB*(a + b - 2*kB + sqroot) /\
#           (2*(k12*(k23 - kB) - (b - kB)*(kB - ku))*sqroot))*tau2
#    p3c = k12*k23*kB/(k12*(k23 - kB) - (b - kB)*(kB - ku))*tau3

    t, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3)
    plt.plot(t, fit, color='k', label=f'3Exp curve')
    t, fit = common_PDF.Exp3(p1c, p2c, tau1c, tau2c, tau3c)
    plt.plot(t, fit, color='b', label=f'3Exp back transformed') 
       
    


    plt.title(f'tau1 {tau1} tau2 {tau2} tau3 {tau3} P1 {p1} P2 {p2} P3 {p3:.2f}')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'time (s)')
