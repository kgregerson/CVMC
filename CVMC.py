from scipy.stats import norm
from math import exp, sqrt, log
import numpy as np

def BlackScholesGamma(S,K,r,div,T,sigma):
    d1 = (np.log(S/K) + (r - div + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    BS_Gamma = np.exp(-div*T)*((norm.pdf(d1))/S*sigma*np.sqrt(T))
    return BS_Gamma
    
def BlackScholesVega(S, K, r, div, T, sigma):
    d2 = (np.log(S / K) + (r - div - sigma * sigma * 0.5) * T)/( sigma * np.sqrt(T))
    BS_Vega = K * exp(-r*T)*norm.pdf(d2)*sqrt(T)
    return BS_Vega
    
def BlackScholesDelta(S,K,R,Div,T,sigma):
    d1 = ((np.log(S/K)) + (R - Div + (sigma^2)/2) * T)/sigma*(sqrt(T))
    BS_Delta = np.exp(-Div*T)*norm.cdf(d1)
    return BS_Delta
    
def Lookback_Option_Pricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = engine.steps
    discount_rate = np.exp(-rate * expiry)
    delta_t = expiry
    z = np.random.normal(size = steps)
    
    nudt = (rate - 0.5 * volatility * volatility) * delta_t
    sidt = volatility * np.sqrt(delta_t)    
    
    spot_t = np.zeros((steps, ))
    payoff_t = np.zeros((steps, ))
    
    for i in range(steps):
        spot_t[i] = spot * nudt + sidt * z[i]
        payoff_t[i] = option.payoff(spot_t[i])
        
    price = discount_rate * payoff_t.max()   
    
    return price
    

    
    
    
