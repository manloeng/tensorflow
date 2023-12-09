# Import with pandas
import pandas as pd
from bitcoin_predict.utils.index import show_graph

# Let's read in our Bitcoin data and parse the dates

df = pd.read_csv("/Users/andrewchung/Downloads/BTC-USD.csv") # parse the date column and tell pandas column 1 is a datetime

pd.to_datetime(df.Date, dayfirst=True)

timesteps = df.Date.tolist()
btc_price = df.Close.tolist()

# Only want closing price for each day
bitcoin_prices = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"}).set_index(df['Date'])
bitcoin_prices.head()

bitcoin_prices.plot(figsize=(10,7), ylabel= ("BTC Price"), title=("Price of Bitcoin from 1 Oct 2013 to 18 May 2021"))

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

show_graph()