import pandas as pd
import matplotlib.pyplot as plt

from bitcoin_predict.utils.index import show_graph
from helper_functions.data_prep.split_data import split_data
from helper_functions.evaluation.plot_graph import plot_time_series

# Let's read in our Bitcoin data and parse the dates
df = pd.read_csv(
    "/Users/andrewchung/Downloads/BTC-USD.csv")  # parse the date column and tell pandas column 1 is a datetime
pd.to_datetime(df.Date, dayfirst=True)

timesteps = df.Date.tolist()
btc_price = df.Close.tolist()

# Only want closing price for each day
bitcoin_prices = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"}).set_index(df['Date'])

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

# splitting dataset
X_train, X_test, y_train, y_test = split_data(timesteps, prices)

# Create a naive forecast
naive_forecast = y_test[:-1]

# Plot naive forecast
fig, ax = plt.subplots(figsize=(10, 8))
# plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(fig=fig, ax=ax, timesteps=X_test, values=y_test, start=600, format="-", label="Test data")
plot_time_series(fig=fig, ax=ax, timesteps=X_test[1:], values=naive_forecast, start=600, format="-",
                 label="Naive Forecast")

show_graph()
