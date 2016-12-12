import abc
import numpy as np
from scipy.stats import binom


class PricingEngine(object, metaclass=abc.ABCMeta):
    """
    An option pricing engine interface.

    """

    @abc.abstractmethod
    def calculate(self):
        """
        A method to implement an option pricing model.

        The pricing method may be either an analytic model (i.e. Black-Scholes or Heston) or
        a numerical method such as lattice methods or Monte Carlo simulation methods.

        """

        pass
    
class MonteCarloPricingEngine(PricingEngine):
    def __init__(self, replications, steps, pricer):
        self.__replications = replications
        self.__steps = steps
        self.__pricer = pricer
        
    @property
    def replications(self):
        return self.__replications
    
    @replications.setter
    def replications(self, new_replications):
        self.__replications = new_replications
        
    @property
    def steps(self):
        return self.__steps
    
    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps
    
    def calculate(self, option, data):
        return self.__pricer(self, option, data)


class BinomialPricingEngine(PricingEngine):
    """
    A concrete PricingEngine class that implements the Binomial model.

    Args:
        

    Attributes:


    """

    def __init__(self, steps, pricer):
        self.__steps = steps
        self.__pricer = pricer

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)
    



def EuropeanBinomialPricer(pricing_engine, option, data):
    """
    The binomial option pricing model for a plain vanilla European option.

    Args:
        pricing_engine (PricingEngine): a pricing method via the PricingEngine interface
        option (Payoff):                an option payoff via the Payoff interface
        data (MarketData):              a market data variable via the MarketData interface

    """

    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
    nodes = steps + 1
    dt = expiry / steps 
    u = np.exp((rate * dt) + volatility * np.sqrt(dt)) 
    d = np.exp((rate * dt) - volatility * np.sqrt(dt))
    pu = (np.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * expiry)
    spotT = 0.0
    payoffT = 0.0
    
    for i in range(nodes):
        spotT = spot * (u ** (steps - i)) * (d ** (i))
        payoffT += option.payoff(spotT)  * binom.pmf(steps - i, steps, pu)  
    price = disc * payoffT 
     
    return price 

def AmericanBinomialPricer(pricing_engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
    nodes = steps + 1
    h = expiry/steps
    u = np.exp((rate * h) + volatility * np.sqrt(h))
    d = np.exp((rate* h) - volatility * np.sqrt(h))
    pu = (np.exp(rate * h) - d)/ (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * expiry)

    V = [[0.0 for k in range(i + 1)] for j in range (steps + 1)]
    
    for i in range(nodes):
        
        V[s][i] = max(spot * (u ** (steps - i)) * (d**(i)) - strike, 0.0)
        
    for i in range(steps - 1, -1, -1):
        for k in range(i + 1):
            V1 = (disc * V[i+1][k+1] + pd * V[i+1][k])
            V2 = max(spot - strike, 0)
            V[i][k] = max(V1,V2)
    return V[0][0]

class LookbackPricingEngine(PricingEngine):
    """
    """

    def __init__(self, steps, pricer):
        self.__steps = steps
        self.__pricer = pricer

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)

def LookbackOptionPricer(pricing_engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
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
    

def BlackScholesDelta(St, t, K, T, sig, r, div):
    tau = T - t
    d1 = (np.log(St/K) + (r - div + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
    delta = np.exp(-div * tau) * norm.cfd(d1)

    return delta


def ControlVariatePricer(engine, option, data):
    S = 100
    K = 100
    sig = 0.20
    r = 0.06
    T = 1.0
    div = 0.03
    N = 10
    M = 100
    beta1 = -1
    dt = T / N
    nudt = (r - div - 0.5 * sig**2) * dt
    sigsdt = sig * sqrt(dt)
    erddt = exp((r - div) * dt)

    sumCT = 0
    sumCT2 = 0

    for j in range(M):

        St = S
        cv = 0

        for i in range(N):
            t = i * dt
            delta = BlackScholesDelta(St, t, K, T, sig, r, div)
            z = np.random.normal(size=1)
            Stn = St * np.exp(nudt + sigsdt * z)
            cv = cv + delta * (Stn - St * erddt)
            St = Stn

        CT = np.maximum(St - K, 0.0) + beta1 * cv
        sumCT += CT
        sumCT2 += CT * CT

    call_value = (sumCT / M) * np.exp(-r * T)
    SD = sqrt((sumCT2 - sumCT * sumCT / M) * exp(-2*r*T) / (M-1))
    return call_value
    
def Naive_Monte_Carlo_Pricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = engine.steps
    replications = engine.replications
    discount_rate = np.exp(-rate * expiry)
    delta_t = expiry / steps
    z = np.random.normal(size = steps)
    
    nudt = (rate - 0.5 * volatility * volatility) * expiry
    sidt = volatility * np.sqrt(expiry)
    
    spot_t = np.zeros((replications, ))
    payoff_t = 0.0
    for i in range(replications):
        spot_t = spot * np.exp(nudt + sidt *z[i])
        
        payoff_t += option.payoff(spot_t)
        
    payoff_t /= replications
    price = discount_rate * payoff_t
    
    return price