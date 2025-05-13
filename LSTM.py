import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

# 2. Feature Engineering
nifty['SMA_20'] = SMAIndicator(close=nifty['Close'], window=20).sma_indicator()
nifty['SMA_50'] = SMAIndicator(close=nifty['Close'], window=50).sma_indicator()
nifty['RSI_14'] = RSIIndicator(close=nifty['Close'], window=14).rsi()
macd = MACD(close=nifty['Close'])
nifty['MACD_diff'] = macd.macd_diff()
atr = AverageTrueRange(high=nifty['High'], low=nifty['Low'], close=nifty['Close'], window=14)
nifty['ATR_14'] = atr.average_true_range()
nifty['Price_Change_%'] = nifty['Close'].pct_change() * 100
nifty['Volatility_5D'] = nifty['Close'].rolling(5).std()
nifty = nifty.dropna()

# 3. Define Features and Target
features = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD_diff', 'ATR_14', 'Volume', 'Price_Change_%', 'Volatility_5D']
X = nifty[features]
y = nifty['Close'].shift(-1).dropna()
X = X.iloc[:-1]  # Align X with y

# 4. Scale the Features and Target using MinMaxScaler for LSTM
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 5. Reshape Data for LSTM (Samples, Timesteps, Features)
# We'll use a window of 60 days (you can experiment with this)
def create_dataset(X, y, time_step=60):
    X_data, y_data = [], []
    for i in range(len(X) - time_step):
        X_data.append(X[i:i + time_step])
        y_data.append(y[i + time_step])
    return np.array(X_data), np.array(y_data)

X_lstm, y_lstm = create_dataset(X_scaled, y_scaled)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, shuffle=False, test_size=0.2)
# 7. Build LSTM Model
model = Sequential()

# First LSTM layer with Dropout and L2 regularization
model.add(LSTM(units=100, return_sequences=True, 
               input_shape=(X_train.shape[1], X_train.shape[2]), 
               kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))

# Second LSTM layer with Dropout
model.add(LSTM(units=100, return_sequences=False, 
               kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error')



# Train the LSTM Model and include validation data to track validation loss
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 9. Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 10. Evaluate
print("R² Score:", r2_score(scaler_y.inverse_transform(y_test), y_pred))
print("RMSE:", np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test), y_pred)))


dates = nifty.index[60 + int(len(X_lstm) * 0.8):]  # Dates corresponding to X_test



# 11. Plot
plt.figure(figsize=(12, 6))
plt.plot(dates, scaler_y.inverse_transform(y_test), label='Actual')
plt.plot(dates, y_pred, label='Predicted')
plt.title("NIFTY 50 Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 Close Price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Predict Next Day
next_input = X_scaled[-60:].reshape(1, 60, X_scaled.shape[1])
next_prediction_scaled = model.predict(next_input)
next_prediction = scaler_y.inverse_transform(next_prediction_scaled)
print("Predicted Next Close:", next_prediction[0][0])
