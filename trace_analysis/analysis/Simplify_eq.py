# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:36:48 2020

@author: pimam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sm
from sympy.solvers.solveset import _transolve as transolve
a, b, c, S, SS= sm.symbols('a b c S SS')
t = sm.symbols('t', positive=True)
s = sm.symbols('s')
ku, kab, kba, kB, kbc = sm.symbols('ku kab kba kB kbc')

#f1 = (s+a)/(s-S)/(s-SS)
#f1 = sm.apart(f1,s)
#print('f1 =', f1)
#
#f2 = (s+a)/(s-S)/(s-SS)/(s+b)
#f2 = sm.apart(f2,s)
#print('f2 =', f2)

f4 = (s+b)/(s-S)/(s-SS)
f4 = sm.apart(f4,s)
print('f4 =', f4)

#f3 = (s+a)*(s+b)/(s-S)/(s-SS)/(s+c)
#f3 = sm.apart(f3,s)
##print('f2 =', f3)

A = s+ku+kab
B = s+kba+kbc+kB
C = s+kB

Roots12 = (A*B - kab*kba)
R12 = sm.solve(Roots12,s)
R1 = R12[0]
R2 = R12[1]
#print('R12=', R12)
print('R1=', R1)
R3 = B - s
R4 = C - s

psi_nom = (B**2)*C*ku + kab*kB*(B+kbc)*(B+C)*A
psi_denom = (s-R1)*(s-R2)*(s-R3)*(s-R4)
psi_fact = sm.apart(psi_nom/psi_denom, s)
print('psi= ',psi_fact)

term1 = B*ku/Roots12
term1smp = sm.apart(term1, s)

#w = sm.symbols('w', real=True)
#expression = s/(s**2+w**2)
P1u = sm.inverse_laplace_transform(term1smp, s, t)
print('P1u= ', P1u)

ku, kab, kba, kB, kbc = sm.symbols('ku kab kba kB kbc')
P, PP, T, TT = sm.symbols('P PP T TT')
sm.solve(1/T+1/TT-(ku+kab+kba+kbc),ku)
ku = -kab - kba - kbc + 1/TT + 1/T
sm.solve(1/T/TT-(ku+kab)*(kba+kbc)+kab*kba, kab)
kab = -kba - 2*kbc - kbc**2/kba + 1/TT + kbc/(TT*kba) + 1/T + kbc/(T*kba) - 1/(T*TT*kba)

sm.solve(kab*kbc-(1-P-PP)*(kB-1/T)*(kB-1/TT),kbc)
kbc = (-P - PP + T*TT*kB**2*(-P - PP + 1) + T*kB*(P + PP - 1) + TT*kB*(P + PP - 1) + 1)/(T*TT*kab)

sm.solve(P-(ku*(kba+kbc-1/T)+kab*kbc*kB/(kB-1/T))*(T**2)*TT/(TT-T),kba)
kba = (-P*T**2*kB + P*T*TT*kB + P*T - P*TT - T**3*TT*kB*kab*kbc - T**3*TT*kB*kbc*ku + T**2*TT*kB*ku + T**2*TT*kbc*ku - T*TT*ku)/(T**2*TT*ku*(T*kB - 1))

sm.solve([1/T+1/TT-(ku+kab+kba+kbc),
          1/T/TT-((ku+kab)*(kba+kbc)-kab*kba),
          kab*kbc-(1-P-PP)*(kB-1/T)*(kB-1/TT),
          P-(ku*(kba+kbc-1/T)+kab*kbc*kB/(kB-1/T))*(T**2)*TT/(TT-T)],[ku,kab,kba,kbc])
s12 = sm.solve(expr2,s)
print('s12=', s12)

(kub (E^(1/
        2 (-k12 - k21 - k23 - kub + 
          Sqrt[(k12 + k21 + k23 + kub)^2 - 
           4 (k12 k23 + (k21 + k23) kub)]) t) (-k12 + k21 + k23 - 
         kub + Sqrt[(k12 + k21 + k23 + kub)^2 - 
          4 (k12 k23 + (k21 + k23) kub)]) + 
      E^(-(1/2) (k12 + k21 + k23 + kub + 
          Sqrt[(k12 + k21 + k23 + kub)^2 - 
           4 (k12 k23 + (k21 + k23) kub)]) t) (k12 - k21 - k23 + kub +
          Sqrt[(k12 + k21 + k23 + kub)^2 - 
          4 (k12 k23 + (k21 + k23) kub)])))/(2 Sqrt[(k12 + k21 + k23 +
        kub)^2 - 4 (k12 k23 + (k21 + k23) kub)]) + 
 k12 k23 kb (E^(-kb t)/(
    k12 (k23 - kb) - (k21 + k23 - kb) (kb - kub)) - (
    E^(1/2 (-k12 - k21 - k23 - kub + 
        Sqrt[(k12 + k21 + k23 + kub)^2 - 
         4 (k12 k23 + (k21 + k23) kub)]) t) (k12 + k21 + k23 - 2 kb + 
       kub + Sqrt[(k12 + k21 + k23 + kub)^2 - 
        4 (k12 k23 + (k21 + k23) kub)]))/(
    2 (k12 (k23 - kb) - (k21 + k23 - kb) (kb - kub)) Sqrt[(k12 + k21 +
         k23 + kub)^2 - 4 (k12 k23 + (k21 + k23) kub)]))

a = ku + k12
b = k21 + k23
sqroot = np.sqrt((a + b)**2 - 4*(k12*k23 + b*ku))

(ku*(np.exp(1/2*(- a - b + sqroot)*t) * (-a + b + sqroot) +\
     np.exp(-(1/2)*(a + b + sqroot)*t) * (a - b + sqroot))) / (2*sqroot) +\
       
k12*k23*kb*(np.exp(-kb*t)/(k12*(k23 - kb) - (b - kb)*(kb - ku)) -\
    (np.exp(1/2*(-a - b + sqroot)*t) * (a + b - 2*kb + sqroot)) /\      
    (2*(k12*(k23 - kb) - (b - kb)*(kb - ku))*sqroot))

