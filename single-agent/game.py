from single_agent_mdp import buy, sell
import pandas as pd
import random
from player import Player
# Define time range (naive datetimes)
time_start = '2024-10-01 00:00:00'
time_end = '2025-04-01 23:59:59'
cash = 10000
portfolio = {}
stocks = ['AAPL']
for stock in stocks:
    portfolio[stock] = 0
file = '../hourly_data/AAPL_hourly.csv'
df = pd.read_csv(file, skiprows=2)
df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert(None)
df = df[(df['Datetime'] >= time_start) & (df['Datetime'] <= time_end)]

player = Player(method='random')

for index, row in df.iterrows():
    current_time = row['Datetime']
    current_price = row['Close']
    action, num_stocks = player.move()
    if action == 'buy':
        portfolio, cash  = buy(num_stocks, current_price,portfolio, cash, 'AAPL')
    elif action == 'sell':
        portfolio, cash  = sell(num_stocks, current_price,portfolio, cash, 'AAPL')
    # print('Current time:', current_time)
    # print('Current price:', current_price)
    # print('Cash:', cash)
    # print('Portfolio:', portfolio)

#SELL ALL REMAINING STOCKS
for stock, quantity in portfolio.items():
    if quantity > 0:
        cash += current_price * quantity
        portfolio[stock] = 0
print(cash)