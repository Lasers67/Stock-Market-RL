stocks = ['AAPL', 'MSFT', 'GOOGL']
import pandas as pd

folder = '../hourly_data/'
data = {}
for stock in stocks:
    file = folder + stock + '_hourly.csv'
    df = pd.read_csv(file, skiprows=2)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert(None)
    data[stock] = df


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and Signal line
    """
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_ema(df, window=200):
    """
    Calculate EMA
    """
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def generate_signals(df):
    df['Trade_Signal'] = 0  # Rename to avoid conflict
    # Buy: MACD crosses above Signal BELOW 0 + Price > EMA
    buy_cond = (
        (df['MACD'] > df['MACD_Signal']) &
        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)) &
        (df['MACD'] < 0) &
        (df['Close'] > df['EMA'])
    )
    df.loc[buy_cond, 'Trade_Signal'] = 1  # Buy signal
    # Short: MACD crosses below Signal ABOVE 0 + Price < EMA
    short_cond = (
        (df['MACD'] < df['MACD_Signal']) &
        (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)) &
        (df['MACD'] > 0) &
        (df['Close'] < df['EMA'])
    )
    df.loc[short_cond, 'Trade_Signal'] = -1  # Short signal
    return df

for stock in stocks:
    df = data[stock]
    df = calculate_macd(df)
    df = calculate_ema(df)
    df = generate_signals(df)
    for i in range(len(df)):
        signal = df['Trade_Signal'].iloc[i]
        if signal == 1:
            print(f"BUY {stock} at {df['Close'].iloc[i]} on {df['Datetime'].iloc[i]}")
        elif signal == -1:
            print(f"SHORT {stock} at {df['Close'].iloc[i]} on {df['Datetime'].iloc[i]}")
    data[stock] = df