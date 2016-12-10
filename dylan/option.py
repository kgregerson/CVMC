class Option(object):
    """
    A facade class to encapculate an option, a pricing method, and market data.

    Attributes:
        payoff (Payoff):        the option payoff function
        engine (PricingEngine): the option pricing method
        data (MarketData):      the market data 

    Methods:
        price: returns a float containing the option price.

    """

    def __init__(self, payoff, engine, data):
        self.__payoff = payoff
        self.__engine = engine
        self.__data = data

    def price(self):
        """
        The option price. 

        Returns a float containing the option price.

        """

        return self.__engine.calculate(self.__payoff, self.__data)
