# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:33:17 2020

@author: pimam
"""
from trace_analysis.analysis import SAfitting_based_on_Z_with_tcut as SA
from trace_analysis.analysis import common_PDF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def PDFExp3(p1, p2, tau1, tau2, tau3, tcut=0, Tmax=1000, log=False):

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


def P2expcut(dwells, params, Tcut, Ncut, tcut):
    P1, tau1, tau2 = params
    Pi = P1/tau1*np.exp(-dwells/tau1)+(1-P1)/tau2*np.exp(-dwells/tau2)
    Pcut = P1*np.exp(-Tcut/tau1)+(1-P1)*np.exp(-Tcut/tau2)
    pcut = P1*np.exp(-tcut/tau1)+(1-P1)*np.exp(-tcut/tau2)
    return Pi, Pcut, pcut


def P3expcut(dwells, params, Tcut, Ncut, tcut):
    P1, P2, tau1, tau2, tau3 = params
    Pi = P1/tau1*np.exp(-dwells/tau1)+P2/tau2*np.exp(-dwells/tau2) + \
        (1 - P1 - P2)/tau3*np.exp(-dwells/tau3)
    Pcut = P1*np.exp(-Tcut/tau1)+P2*np.exp(-Tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-Tcut/tau3)
    pcut = P1*np.exp(-tcut/tau1)+P2*np.exp(-tcut/tau2) + \
        (1 - P1 - P2)*np.exp(-tcut/tau3)
#    print('Pcut ', Pcut)
    return Pi, Pcut, pcut


def BIC(dwells, k, LLike):
    bic = np.log(dwells.size)*k + 2*LLike
    return bic


def LogLikelihood(dwells, params, model, Tcut, Ncut, tcut):
    Pi, Pcut, pcut = model(dwells, params, Tcut, Ncut, tcut)
    LLikecut = -Ncut * np.log(Pcut)
    LLike = -np.sum(np.log(Pi)) + LLikecut + np.log(pcut-Pcut)
#    Likecut = Pcut**Ncut
#    Like = np.prod(Pi)*Likecut/pcut
    print('Dpcut', pcut-Pcut)
    return LLike


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

bsize =0.02
bin_edges = 10**(np.arange(np.log10(min(dataout)), np.log10(max(dataout)) + bsize, bsize))
values, bins = np.histogram(dataout, bins=bin_edges, density=True)
centers = (bins[1:] * bins[:-1])**0.5
plt.plot(centers, values, '.', color='r')
p1=0.52
p2=0.00
tau1=2.2
#tau2=20
#tau3=20
tau2=20
tau3=81
delta_p = 0.04
delta_t = 0.05
#Z1 = np.log(p1/(1-p1-p2))
#Z2= np.log(p2/(1-p1-p2))
#T1 = np.log(tau1)
#T2 = np.log(tau2)
#T3 = np.log(tau3)
Tcut = dataout.max()
model = P3expcut
time, fit = common_PDF.Exp3(p1, p2, tau1, tau2, tau3, tcut=0.9, Tmax=Tcut, log=True)
l, = plt.loglog(time, fit, label='fit1')

ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'
axP1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axP2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axTau1 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
#axtau2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#axtau3 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

sP1= Slider(axP1, 'P1', 0, 1.0, valinit=p1, valstep=delta_p)
sP2 = Slider(axP2, 'P2', 0, 1.0, valinit=p2, valstep=delta_p)
sTau1 = Slider(axTau1, 'tau1', 0, 5.0, valinit=tau1, valstep=delta_t)
#stau2 = Slider(axtau1, 'tau2', 0, 1.0, valinit=tau2, valstep=delta_t)
#stau3 = Slider(axtau1, 'tau3', 0, 1.0, valinit=tau3, valstep=delta_t)

def update(val):
    P1 = sP1.val
    P2 = sP2.val
    Tau1 = sTau1.val
#    Tau2 = stau2.val
#    Tau3 = stau3.val
    LL = LogLikelihood(dataout, [P1, P2, Tau1, tau2, tau3], model, Tcut, 0, tcut=0.9)
    bic = BIC(dataout, 5, LL)
    print(f'LL: {LL} BIC: {bic}')
    time, fit = PDFExp3(P1, P2, Tau1, tau2, tau3, tcut=0.9, Tmax=Tcut, log=True)
    l.set_ydata(fit)
    fig.canvas.draw_idle()


sP1.on_changed(update)
sP2.on_changed(update)
sTau1.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sP1.reset()
    sP1.reset()
button.on_clicked(reset)

#rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


#def colorfunc(label):
#    l.set_color(label)
#    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)

plt.show()