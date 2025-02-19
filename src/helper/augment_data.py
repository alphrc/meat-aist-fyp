import os
import numpy as np
import pandas as pd
import ta

directory = './data/Stock'
file_names = os.listdir(directory)
tickers = [file_name.split('_')[0] for file_name in file_names]

for ticker in tickers:
    # Read the raw data file
    raw_data_file = f'./data/raw/{ticker}.csv'
    raw_data = pd.read_csv(raw_data_file)

    # Calculate the natural logarithm of OHLC
    raw_data['Log_Open'] = np.log(raw_data['Open'])
    raw_data['Log_High'] = np.log(raw_data['High'])
    raw_data['Log_Low'] = np.log(raw_data['Low'])
    raw_data['Log_Close'] = np.log(raw_data['Close'])
    raw_data['Log_Volume'] = np.log(raw_data['Volume'])

    # Calculate daily price percentage change
    raw_data['Price_Change'] = raw_data['Close'].pct_change()

    # Calculate price change direction
    raw_data['Direction'] = raw_data['Price_Change'].apply(lambda x: 1 if x > 0 else -1)

    # Calculate RSI
    raw_data['RSI_7'] = ta.momentum.RSIIndicator(raw_data['Close'], window=7).rsi()
    raw_data['RSI_14'] = ta.momentum.RSIIndicator(raw_data['Close'], window=14).rsi()
    raw_data['RSI_21'] = ta.momentum.RSIIndicator(raw_data['Close'], window=21).rsi()

    # Calculate EMA
    raw_data['EMA_9'] = raw_data['Close'].ewm(span=9, adjust=False).mean()
    raw_data['EMA_21'] = raw_data['Close'].ewm(span=21, adjust=False).mean()
    raw_data['EMA_55'] = raw_data['Close'].ewm(span=55, adjust=False).mean()

    # Save augmented data to a new file
    augmented_data_file = f'./data/augmented/{ticker}.csv'
    raw_data.to_csv(augmented_data_file, index=False)