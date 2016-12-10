from dylan.payoff import ExoticPayoff, call_payoff, put_payoff
from dylan.engine import LookbackPricingEngine, LookbackOptionPricer
from dylan.marketdata import MarketData
from dylan.option import Option

def main():
    spot = 10
    strike = 100
    rate = 0.06
    volatility = 0.20
    #reps = 1000
    expiry = 1.0
    steps = 52
    dividend = 0.03

    the_call = ExoticPayoff(expiry, strike, call_payoff)
    Convar_LBO = LookbackPricingEngine(steps, LookbackOptionPricer)
    the_data = MarketData(rate, spot, volatility, dividend)

    the_option = Option(the_call, Convar_LBO, the_data)
    fmt = "The call option price is {0:0.3f}"
    print(fmt.format(the_option.price()))


if __name__ == "__main__":
    main()
