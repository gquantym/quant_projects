import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

asset = 'APPL'
# Downloading Historical Data
ticker = 'AAPL'  # You can change this to any stock
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Code for calculating the Rolling Mean and Standard Deviation
data['Rolling Mean'] = data['Close'].rolling(window=20).mean()  # 20-day rolling mean
data['Rolling Std'] = data['Close'].rolling(window=20).std()  # 20-day rolling standard deviation

# Defining Upper and Lower Boundaries
data['Upper Bound'] = data['Rolling Mean'] + (2 * data['Rolling Std'])
data['Lower Bound'] = data['Rolling Mean'] - (2 * data['Rolling Std'])



# Ensure 'Close' is a Series instead of a DataFrame
data['Close'] = data['Close'].squeeze()  
close = data['Close'].squeeze()

# Generating Buy and Sell Signals
data['Signal'] = np.where(close < data['Lower Bound'], 1, 0)  # Buy signal
#print("Test 1")
data['Signal'] = np.where(close > data['Upper Bound'], -1, data['Signal'])  # Sell signal
#print("Test 2")


# Backtest the Strategy
data['Position'] = data['Signal'].shift()  # Shift to avoid buying/selling on the same day
data['Returns'] = data['Close'].pct_change()
data['Strategy Returns'] = data['Position'] * data['Returns']

# Plotting the results !!
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label=f'{ticker} Close Price', alpha=0.5)
plt.plot(data['Rolling Mean'], label='20-day Rolling Mean', linestyle='--')
plt.plot(data['Upper Bound'], label='Upper Bound', linestyle='--')
plt.plot(data['Lower Bound'], label='Lower Bound', linestyle='--')
plt.title('Mean Reversion Strategy - {}'.format(asset))
plt.legend(loc='best')
plt.show()

# Cumulative Returns i.e How Much Do We Make (Compound) and Plot
cumulative_returns = (1 + data['Strategy Returns']).cumprod()
cumulative_returns.plot(title='Cumulative Strategy Returns - {}'.format(asset))
plt.show()
