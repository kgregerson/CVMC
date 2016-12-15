from dylan.payoff import ExoticPayoff, call_payoff
from dylan.engine import ControlVariateMCPricer, ControlVariateEngine
from dylan.marketdata import MarketData
from dylan.option import Option

def main():
    spot = 100
    strike = 100
    rate = 0.06
    volatility = 0.20
    expiry = 1.0
    steps = 10
    replications = 100
    dividend = 0.03
    alpha = 5
    Vbar = .02
    xi = 52

    the_call = ExoticPayoff(expiry, strike, call_payoff)
    the_mcpe = ControlVariateEngine(steps, replications, alpha, Vbar, xi, ControlVariateMCPricer)
    the_data = MarketData(rate, spot, volatility, dividend)

    the_option = Option(the_call, the_mcpe, the_data)
    fmt = "The call option price is {0:0.3f}"
    print(fmt.format(the_option.price()))


if __name__ == "__main__":
    main()

