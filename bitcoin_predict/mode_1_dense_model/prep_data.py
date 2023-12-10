import tensorflow as tf

from bitcoin_predict.utils.index import get_bitcoin_prices
from helper_function.data_prep.time_series_data_prep import get_labelled_windows, make_windows, make_train_test_splits


def prep_data_model_2(window_size=7, horizon=1):
    bitcoin_prices = get_bitcoin_prices()

    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()

    test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(8), axis=0))
    print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")

    full_windows, full_labels = make_windows(prices, window_size, horizon)
    len(full_windows), len(full_labels)

    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    len(train_windows), len(test_windows), len(train_labels), len(test_labels)

    return train_windows, test_windows, train_labels, test_labels
