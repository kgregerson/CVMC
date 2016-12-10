from scipy.stats import norm
from math import exp, sqrt, log
import numpy as np

def BlackScholesGamma(S,K,r,div,T,sigma):
    d1 = (np.log(S/K) + (r - div + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    BS_Gamma = np.exp(-div*T)*((norm.pdf(d1))/S*sigma*np.sqrt(T))
    return BS_Gamma
    
def BlackScholesVega(S, K, r, div, T, sigma):
    d2 = (np.log(S / K) + (r - div - sigma * sigma * 0.5) * T)/( sigma * np.sqrt(T))
    vega = K * exp(-r*T)*norm.pdf(d2)*sqrt(T)
    return vega
    
def BlackScholesDelta(S,K,R,Div,T,sigma):
    d1 = ((np.log(S/K)) + (R - Div + (sigma^2)/2) * T)/sigma*(sqrt(T))
    Delta = np.exp(-Div*T)*norm.cdf(d1)
    return Delta
    
    
    
    
