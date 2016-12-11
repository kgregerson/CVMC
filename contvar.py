import numpy as np
from scipy.stats import norm
from math import sqrt, exp, log


def BlackScholesDelta(St, t, K, T, sig, r, div):
    tau = T - t
    d1 = (np.log(St/K) + (r - div + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
    delta = np.exp(-div * tau) * norm.cdf(d1)

    return delta
    
def BlackScholesGamma(St, t, K, T, sig, r, div):
    tau = T - t
    d1 = (np.log(St/K) + (r - div + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
    gamma = np.exp(-div * tau ) * (norm.pdf(d1) / St * sig * np.sqrt(tau))
    return gamma
    
def BlackScholesVega(St, t, K, T, sig, r, div):
    tau = T - t
    d2 = (np.log(S / K) + (r - div - sig * sig * 0.5) * tau)/( sig * np.sqrt(tau))
    vega = K * exp(-r * tau) * norm.pdf(d2) * np.sqrt(tau)
    return vega
    
## main
S = 100
K = 100
sig = 0.20
r = 0.06
T = 1
div = 0.03
alpha = 5
Vbar = .02
xi = 52
N = 10
M = 100
beta1 = -1
beta2 = -1
beta3 = -1

dt = T / N
nudt = (r - div - 0.5 * sig**2) * dt
sigsdt = sig * sqrt(dt)
xisdt = xi * np.sqrt(dt)
erddt = exp((r - div) * dt)
egam1 = exp(2*(r-div)*dt)
egam2 = -2*erddt+1
eveg1 = exp(-alpha*dt)
eveg2 = Vbar - Vbar*eveg1

sumCT = 0
sumCT2 = 0

for j in range(M):
    Vt = sig * sig
    St = S
    maxSt = S
    cv1 = 0
    cv2 = 0
    cv3 = 0
    
    s = np.zeros(N)
    v = np.zeros(N)

    s[0] = S
    v[0] = Vbar

    z1 = np.random.normal(size=N)
    z2 = np.random.normal(size=N)
    
    for i in range(N):
        t = (i-1) * dt
        delta = BlackScholesDelta(s[i], t, K, T, sig, r, div)
        gamma = BlackScholesGamma(s[i], t, K, T, sig, r, div)
        vega = BlackScholesVega(s[i], t, K, T, sig, r, div)
        
        ##### Evolve Variance #####
        v[i] = v[i-1] + alpha * dt + (Vbar - v[i-1]) + xisdt * z1[i]
        if v[i] < 0.0: 
            v[i] = 0.0
        
        #####Evolve Asset Price #####
        s[i] = s[i-1] * np.exp((r - 0.5 * v[i-1]) * dt + np.sqrt(v[i-1]) * np.sqrt(dt) * z2[i])
        
        ####### Accumulate Control Variates ######
        cv1 = cv1 + delta * (s[i] - s[i-1] * erddt)
        cv2 = cv2 + gamma * ((s[i] - s[i-1]) * (s[i] - s[i-1]) - s[i-1] * s[i-1] * (egam1 * exp(v[i-1]*dt) + egam2))
        cv3 = cv3 + vega * ((v[i]- v[i-1])-(Vt * eveg1 + eveg2 - v[i-1]))
    
        smax = np.amax(s)
        
    CT = np.maximum(smax - K, 0.0) + beta1 * cv1 + beta2 * cv3 + beta3 * cv3
    sumCT += CT
    sumCT2 += CT * CT

call_value = (sumCT / M) * np.exp(-r * T)
SD = sqrt((sumCT2 - sumCT * sumCT / M) * exp(-2*r*T) / (M-1))

print(call_value, SD)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:03:55 2016

@author: Garrett
"""

