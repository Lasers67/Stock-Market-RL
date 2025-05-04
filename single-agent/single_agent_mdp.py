actions = ['buy', 'sell', 'hold']
def buy(num_stocks, price, potfolio,in_hand_cash, stock):
    """
    Buy stock with the given amount of money.
    """
    old_potfolio = potfolio.copy()
    if num_stocks <= 0:
        return potfolio, in_hand_cash
    if price <= 0:
        return potfolio, in_hand_cash
    if potfolio[stock] < 0:
        return potfolio, in_hand_cash
    if in_hand_cash < num_stocks * price:
        return potfolio, in_hand_cash
    old_potfolio[stock] += num_stocks
    in_hand_cash -= num_stocks * price
    return old_potfolio, in_hand_cash

def sell(num_stocks, price, potfolio, in_hand_cash, stock):
    """
    Sell stock with the given amount of money.
    """
    old_potfolio = potfolio.copy()
    if num_stocks <= 0:
        return potfolio, in_hand_cash
    if price <= 0:
        return potfolio, in_hand_cash
    if potfolio[stock] < num_stocks:
        return potfolio, in_hand_cash
    old_potfolio[stock] -= num_stocks
    in_hand_cash += num_stocks * price
    return old_potfolio, in_hand_cash