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


def Internal_rates_3state_model(P1, tau1, tau2):
#A double exponential binding time distribution can be described by a 3 state model
    k21 = 1/(tau1*(1-P1) + tau2*P1)
    k13 = P1/tau1 + (1-P1)/tau2
    k12 = (1-P1)/tau1 + P1/tau2 - k21

    return k21, k13, k12

def Error_rates__3states_from_bootstrap(boot_data):
    k21, k13, k12 = Internal_rates_3state_model(boot_data)