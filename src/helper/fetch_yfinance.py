import os
import yfinance as yf

directory = './data/Stock'
file_names = os.listdir(directory)
tickers = [file_name.split('_')[0] for file_name in file_names]

start_date = '2013-01-01'
end_date = '2022-12-31'

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    csv_filename = f'./data/raw
    /{ticker}.csv'
    data.to_csv(csv_filename)