import pandas as pd
import matplotlib.pyplot as plt

from bitcoin_predict.utils.index import show_graph, get_bitcoin_prices
from helper_functions.data_prep.split_data import split_data
from helper_functions.evaluation.plot_graph import plot_time_series
from helper_functions.evaluation.metrics import evaluate_preds

bitcoin_prices = get_bitcoin_prices()

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


naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)

# show_graph()
# should append results onto csv, so it's easy to access
print(naive_results)

# results:
# {'mae': 581.24585, 'mse': 843774.7, 'rmse': 918.5721, 'mape': 1.9395278, 'mase': 1.0008947}