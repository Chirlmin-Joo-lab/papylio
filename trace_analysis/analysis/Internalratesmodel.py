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


def Internal_rates_3state_model(P1, tau1, tau2, stdP1, stdtau1, stdtau2):
#A double exponential binding time distribution can be described by a 3 state model
    k21 = 1/(tau1*(1-P1) + tau2*P1)
    k13 = P1/tau1 + (1-P1)/tau2
    k12 = (1-P1)/tau1 + P1/tau2 - 1/(tau1*(1-P1) + tau2*P1)
    
    stdk21 = (k21**2)*np.sqrt(((1-P1)*stdtau1)**2 + (P1*stdtau2)**2 + ((tau2-tau1)*stdP1)**2)

    return k21, k13, k12
