from single_agent_mdp import buy, sell
import pandas as pd
import random
from player import Player
import smtplib
from email.mime.text import MIMEText
# Define time range (naive datetimes)
time_end = '2025-05-01 23:59:59'
cash = 10000
portfolio = {}
stocks = ['AAPL']

#BUGS
# RL AGENT ONLY SENDING 5 BUY/SELL
 




# Read and preprocess
file = '../hourly_data/AAPL_hourly.csv'
df = pd.read_csv(file, skiprows=2)
df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_convert(None)

# Extract date and sort
df['Date'] = df['Datetime'].dt.date
unique_dates = sorted(df['Date'].unique(), reverse=True)
date_to_episode = {date: i+1 for i, date in enumerate(unique_dates)}
df['episode'] = df['Date'].map(date_to_episode)

# Create time steps within each episode
df = df.sort_values(['Date', 'Datetime'])  # Ensure it's sorted by date and time
df['time'] = df.groupby('Date').cumcount() + 1

# Pad each day to 7 rows if needed
padded_rows = []
for date in unique_dates:
    day_df = df[df['Date'] == date]
    while len(day_df) < 7:
        # Duplicate the last row and adjust time
        last_row = day_df.iloc[-1].copy()
        last_row['time'] = len(day_df) + 1
        day_df = pd.concat([day_df, pd.DataFrame([last_row])], ignore_index=True)
    padded_rows.append(day_df)

# Combine all padded days
df = pd.concat(padded_rows, ignore_index=True)
time_start = df.loc[(df['episode'] == 491) & (df['time'] == 1), 'Datetime'].values[0]
df_test = df[(df['Datetime'] >= time_start) & (df['Datetime'] <= time_end)]
df_test = df_test.sort_values(by='Datetime', ascending=True)


#INITIALIZE PORTFOLIO
for stock in stocks:
    portfolio[stock] = 0
init_state = (df[(df['episode'] == 1) & (df['time'] == 1)]['Close'].values[0],portfolio, cash)
player = Player(method='RL', init_state=init_state, data=df)
def send_email(action, num_stocks, price, time):
    sender = "datastockmarket5@gmail.com"          # Your Gmail address
    app_password = "Data123456"        # Use App Password if 2FA is on
    recipient = "datastockmarket5@gmail.com"             # Receiver's email

    subject = f"Stock Action: {action}"
    body = f"{action} {num_stocks} stocks at {price} on {time}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)

#send_email('buy', 10, 150, '2024-10-01 00:00:00')  # Example usage
cash = 10000
portfolio = {'AAPL': 0}
for index, row in df_test.iterrows():
    current_time = row['Datetime']
    current_price = row['Close']
    action, num_stocks = player.move((current_price, portfolio, cash))
    print(action, num_stocks)
    if action == 'buy':
        portfolio, cash  = buy(num_stocks, current_price,portfolio, cash, 'AAPL')
        # send email to notify the user
        #send_email('buy', num_stocks, current_price, current_time)

    elif action == 'sell':
        portfolio, cash  = sell(num_stocks, current_price,portfolio, cash, 'AAPL')
        #send_email('sell', num_stocks, current_price, current_time)
    print('Current time:', current_time)
    print('Current price:', current_price)
    print('Cash:', cash)
    print('Portfolio:', portfolio)
    init_state = (current_price, portfolio, cash)
#SELL ALL REMAINING STOCKS
for stock, quantity in portfolio.items():
    if quantity > 0:
        cash += current_price * quantity
        portfolio[stock] = 0
print(cash)