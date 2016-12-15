import abc
import numpy as np
from scipy.stats import binom, norm
from math import sqrt, exp, log


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
    
class ControlVariateEngine(PricingEngine):
    def __init__(self, replications, steps, alpha, Vbar, xi, pricer):
        self.__replications = replications
        self.__steps = steps
        self.__pricer = pricer
        self.__alpha = alpha
        self.__Vbar = Vbar
        self.__xi = xi
        
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
    
    @property
    def alpha(self):
        return self.__alpha
    
    @alpha.setter
    def alpha(self, new_alpha):
        self.__alpha = new_alpha
    
    @property
    def Vbar(self):
        return self.__Vbar
    
    @Vbar.setter
    def Vbar(self, new_Vbar):
        self.__Vbar = new_Vbar
        
    @property
    def xi(self):
        return self.__xi
    
    @xi.setter
    def xi(self, new_xi):
        self.__xi = new_xi
    
    def calculate(self, option, data):
        return self.__pricer(self, option, data)

def BlackScholesDelta(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - dividend + 0.5 * volatility * volatility) * tau) / (volatility * np.sqrt(tau))
    delta = np.exp(-dividend * tau) * norm.cdf(d1)

    return delta
    
def BlackScholesGamma(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - dividend + 0.5 * volatility * volatility) * tau) / (volatility * np.sqrt(tau))
    gamma = np.exp(-dividend * tau ) * (norm.pdf(d1) / spot * volatility * np.sqrt(tau))
    return gamma
    
def BlackScholesVega(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d2 = (np.log(spot / strike) + (rate - dividend - volatility * volatility * 0.5) * tau)/( volatility * np.sqrt(tau))
    vega = strike * exp(-rate * tau) * norm.pdf(d2) * np.sqrt(tau)
    return vega

    
def ControlVariateMCPricer(engine, option, data):
    beta1 = -1
    beta2 = -1
    beta3 = -1
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = engine.steps
    replications = engine.replications
    alpha = engine.alpha
    Vbar = engine.Vbar
    xi = engine.xi
    dt = expiry / steps
    xisdt = xi * np.sqrt(dt)
    erddt = exp((rate - dividend) * dt)
    egam1 = exp(2*(rate - dividend)*dt)
    egam2 = -2*erddt+1
    eveg1 = exp(-alpha*dt)
    eveg2 = Vbar - Vbar*eveg1
    
    sumCT = 0
    sumCT2 = 0
    
    for j in range(replications):
        Vt = volatility * volatility
        St = spot
        maxSt = spot
        cv1 = 0
        cv2 = 0
        cv3 = 0
        
        s = np.zeros(steps)
        v = np.zeros(steps)
        
        s[0] = spot
        v[0] = Vbar
        
        z1 = np.random.normal(size=steps)
        z2 = np.random.normal(size=steps)
        
        for i in range(steps):
            t = (i-1) * dt
            delta = BlackScholesDelta(s[i], t, strike, expiry, volatility, rate, dividend)
            gamma = BlackScholesGamma(s[i], t, strike, expiry, volatility, rate, dividend)
            vega = BlackScholesVega(s[i], t, strike, expiry, volatility, rate, dividend)
            
            ##### Evolve Variance #####
            v[i] = v[i-1] + alpha * dt + (Vbar - v[i-1]) + xisdt * z1[i]
            if v[i] < 0.0: 
                v[i] = 0.0
        
            #####Evolve Asset Price #####
            s[i] = s[i-1] * np.exp((rate - 0.5 * v[i-1]) * dt + np.sqrt(v[i-1]) * np.sqrt(dt) * z2[i])
        
            ####### Accumulate Control Variates ######
            cv1 = cv1 + delta * (s[i] - s[i-1] * erddt)
            cv2 = cv2 + gamma * ((s[i] - s[i-1]) * (s[i] - s[i-1]) - s[i-1] * s[i-1] * (egam1 * exp(v[i-1]*dt) + egam2))
            cv3 = cv3 + vega * ((v[i]- v[i-1])-(Vt * eveg1 + eveg2 - v[i-1]))
    
            smax = np.amax(s)
        
        CT = np.maximum(smax - strike, 0.0) + beta1 * cv1 + beta2 * cv3 + beta3 * cv3
        sumCT += CT
        sumCT2 += CT * CT

        call_value = (sumCT / replications) * np.exp(-rate * expiry)
        SD = sqrt((sumCT2 - sumCT * sumCT / replications) * exp(-2*rate*expiry) / (replications-1))

        print(call_value, SD)
    
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