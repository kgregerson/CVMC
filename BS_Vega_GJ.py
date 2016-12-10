import numpy as np
from scipy.stats import norm
from math import exp, sqrt, log

def BlackScholesVega(S, K, r, div, T, sigma):
    d2 = (np.log(S / K) + (r - div - sigma * sigma * 0.5) * T)/( sigma * np.sqrt(T))
    vega = K * exp(-r*T)*norm.pdf(d2)*sqrt(T)
    return vega
    
def ControllVariateMC(K, T, S, sig, r, div, alpha, Vbar, xi, N, M):
    
    dt=1/xi
    sig2 = sig*sig
    alphadt=alpha*dt
    xisdt = xi * np.sqrt(dt)
    erddt = exp((r-div)*dt)
    egam1 = exp(2*(r-div)*dt)
    egam2 = -2*erddt+1
    eveg1 = exp(-alpha*dt)
    eveg2 = Vbar - Vbar*eveg1
    
    sum_CT = 0
    sum_CT2 = 0
    
    beta1 = -.88
    beta2 = -.42
    beta3 = -.0003
    
    for j in range(N): #for each simulation
        St1 = S
        St2 = S
        Vt = sig2
        maxSt1 = St
        maxSt2 = St
        cv1 = 0
        cv2 = 0
        cv3 = 0
        
        for i in range(xi*T): #for each time step
            

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:22:59 2016

@author: Garrett
"""

"""

"""