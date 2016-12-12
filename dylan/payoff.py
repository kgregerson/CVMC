import abc
from numpy import maximum

class Payoff(object, metaclass=abc.ABCMeta):
    """
    An option payoff interface. 

    """

    @property 
    @abc.abstractmethod
    def expiry(self):
        pass

    @expiry.setter
    @abc.abstractmethod
    def expiry(self):
        pass

    @abc.abstractmethod
    def payoff(self):
        pass


class VanillaPayoff(Payoff):
    """
    A plain vanilla option payoff.     

    Args:
        expiry (float):    the option's expiration date.
        strike (int):      the option's strike price.
        payoff (function): the option's payoff function (via the strategy pattern)

    """

    def __init__(self, expiry, strike, payoff):
        self.__expiry = expiry
        self.__strike = strike
        self.__payoff = payoff

    @property
    def expiry(self):
        """
        The option's expiry.

        """
        return self.__expiry

    @expiry.setter
    def expiry(self, new_expiry):
        self.__expiry = new_expiry

    @property
    def strike(self):
        """
        The option's strike.

        Args:
            new_Strike: a new strike price

        """

        return self.__strike

    @strike.setter
    def strike(self, new_strike):
        self.__strike = new_strike

    def payoff(self, spot):
        return self.__payoff(self, spot)
        
class ExoticPayoff(Payoff):
    def __init__(self, expiry, strike, payoff):
        self.__expiry = expiry
        self.__strike = strike
        self.__payoff = payoff
        
    @property
    def expiry(self):
        return self.__expiry

    @expiry.setter
    def expiry(self, new_expiry):
        self.__expiry = new_expiry
    
    @property 
    def strike(self):
        return self.__strike
    
    @strike.setter
    def strike(self, new_strike):
        self.__strike = new_strike
        
    def payoff(self, spot):
        return self.__payoff(self, spot)


def call_payoff(option, spot):
    """
    The payoff function for a European call option. 

    Args:
        option:       the self variable from the Payoff class that aggregates the function.
        spot (float): the price of the underlying asset

    """

    return maximum(spot - option.strike, 0.0)


def put_payoff(option, spot):
    """
    The payoff function for a European put option.

    Args:
        option:       the self variable from the payoff class that aggregates the function.
        spot (float): the price of the uncerlying asset.
    """

    return maximum(option.strike - spot, 0.0)


