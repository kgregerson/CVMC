# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:03:43 2016

@author: colbygranstrom
"""
from scipy.stats import norm
from math import exp, sqrt, log
import numpy as np
def blackscholesdelta(S,K,R,Div,T,sigma):
    d1 = ((np.log(S/k)) + (R - Div + (sigma^2)/2) * T)/sigma*(sqrt(T))
    Delta = np.exp(-Div*T)*norm.cdf(d1)
    return Delta
    