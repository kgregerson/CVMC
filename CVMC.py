from scipy.stats import norm
from math import exp, sqrt, log
import numpy as np

def BlackScholesGamma(S,K,r,div,T,sigma):
    d1 = (np.log(S/K) + (r - div + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    BS_Gamma = np.exp(-div*T)*((norm.pdf(d1))/S*sigma*np.sqrt(T))
    return BS_Gamma
    
def BlackScholesVega(S, K, r, div, T, sigma):
    d2 = (np.log(S / K) + (r - div - sigma * sigma * 0.5) * T)/( sigma * np.sqrt(T))
    BS_Vega = K * exp(-r*T)*norm.pdf(d2)*np.sqrt(T)
    return BS_Vega
    
def BlackScholesDelta(S,K,r,div,T,sigma):
    d1 = ((np.log(S/K)) + (r - div + (sigma*sigma)/2) * T)/sigma*(np.sqrt(T))
    BS_Delta = np.exp(-div*T)*norm.cdf(d1)
    return BS_Delta
"""   
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
"""   

def ControlVariateMC(K, T, S, sigma, r, div, alpha, Vbar, xi, N, M):
    
    dt=1/xi
    sig2 = sigma*sigma
    alphadt=alpha*dt
    xisdt = xi * np.sqrt(dt)
    sdt = np.sqrt(dt)
    erddt = exp((r-div)*dt)
    egam1 = exp(2*(r-div)*dt)
    egam2 = -2*erddt+1
    eveg1 = exp(-alpha*dt)
    eveg2 = Vbar - Vbar*eveg1
    
    sum_CT = 0
    sum_CT2 = 0
    
    beta1 = 1
    beta2 = 1
    beta3 = 1
    
    for j in range(N): #for each simulation
        St1 = S
        Vt = sig2
        maxSt1 = S
        cv1 = 0
        cv2 = 0
        cv3 = 0
        
        for i in range(xi*T): #for each time step
            t = (i-1) * dt
            delta1=BlackScholesDelta(S,K,r,div,t,sigma)
            gamma1=BlackScholesGamma(S,K,r,div,t,sigma)
            vega1=BlackScholesVega(S,K,r,div,t,sigma)
            
            
            ####### Evolve variance ######
            e = np.random.normal(0,1)
            Vtn = Vt + alphadt * (Vbar-Vt) + xisdt * sigma * e
            
            ####### evolve asset price #######
            e = np.random.normal(0,1)
            Stn1 = St1 * exp((r-div-.5*Vt)*dt + sigma * sdt * e)
            
    
            ####### Accumulate Control variates #####
            cv1 = cv1 + delta1 * (Stn1-St1*erddt)
            cv2 = cv2 + gamma1 * ((Stn1-St1)*(Stn1-St1)-St1*St1*(egam1*exp(Vt*dt)+egam2))
            cv3 = cv3 + vega1 * ((Vtn-Vt)-(Vt*eveg1+eveg2-Vt))
            
            Vt = Vtn
            St1 = Stn1
            
            if(St1 > maxSt1):
                maxSt1 = St1
                
        CT = max(0, maxSt1-K) + beta1*cv1 + beta2*cv2 + beta3*cv3
        
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 + CT * CT
    
    call_value = sum_CT/M*exp(-r*T)
    SD = sqrt((sum_CT2 - sum_CT*sum_CT/M)* exp(-2*r*T)/(M-1))
    SE = SD/sqrt(M)
    
    print(call_value, SD, SE)    

ControlVariateMC(100, 1, 100, .2, .06, .03, 5, .02, 52, 1000, 17.729)

