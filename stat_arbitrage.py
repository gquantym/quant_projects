import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Downloading data
stock1 = 'GOOGL'  # Google
stock2 = 'IBM'   # Facebook
data1 = yf.download(stock1, start='2020-01-01', end='2023-01-01')['Close'].squeeze()
data2 = yf.download(stock2, start='2020-01-01', end='2023-01-01')['Close'].squeeze()

# Code to calculate spread (price difference)
spread = data1 - data2
spread_mean = spread.rolling(window=20).mean()
spread_std = spread.rolling(window=20).std()

# Generate buy and sell signals based on spread 
long_signal = (spread < (spread_mean - 2 * spread_std)).astype(int)  # Buy signal
short_signal = (spread > (spread_mean + 2 * spread_std)).astype(int)  # Sell signal

# Backtesting the Pairs Trading Strategy:
position_stock1 = np.zeros_like(data1)  # Initialize positions
position_stock1[long_signal == 1] = 1   # Buy signal for stock1
position_stock1[short_signal == 1] = -1 # Short signal for stock1

# Opposite position for stock2
position_stock2 = -position_stock1

# Step 5: Calculate Daily Returns for Each Stock (as 1D arrays)
returns_stock1 = data1.pct_change().fillna(0)
returns_stock2 = data2.pct_change().fillna(0)

# Step 6: Calculate Strategy Returns (ensure proper dimension alignment)
strategy_returns = (position_stock1 * returns_stock1) + (position_stock2 * returns_stock2)

# Step 7: Plot the Spread
plt.figure(figsize=(10, 5))
plt.plot(spread, label='Spread', alpha=0.5)
plt.plot(spread_mean, label='Mean', linestyle='--')
plt.plot(spread_mean + 2 * spread_std, label='Upper Bound', linestyle='--')
plt.plot(spread_mean - 2 * spread_std, label='Lower Bound', linestyle='--')
plt.title('Pairs Trading Strategy - {0} vs {1}'.format(stock1,stock2))
plt.xlabel('Date') 
plt.ylabel('Spread (Price Difference in USD)')
plt.legend(loc='best')


#-----------------------------------------

# Step 1: Cumulative Strategy Returns
cumulative_returns = (1 + strategy_returns).cumprod()

# Step 2: Plot Cumulative Returns
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label='Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Pairs Trading Strategy Returns')
plt.legend(loc='best')
plt.show()

# Step 3: Plot Daily Returns Distribution (Histogram)
plt.figure(figsize=(10, 5))
plt.hist(strategy_returns, bins=50, alpha=0.7, color='blue')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Returns')
plt.show()

# Step 4: Plot Rolling Volatility (20-day Standard Deviation of Returns)
rolling_volatility = strategy_returns.rolling(window=20).std()

plt.figure(figsize=(10, 5))
plt.plot(rolling_volatility, label='Rolling Volatility (20-day)', color='red')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Rolling 20-day Volatility of Strategy Returns')
plt.legend(loc='best')
plt.show()