import matplotlib.pyplot as plt
import pandas as pd


def show_graph():
    plt.show()


def get_bitcoin_data():
    # Let's read in our Bitcoin data and parse the dates
    df = pd.read_csv(
        "/Users/andrewchung/Downloads/BTC-USD.csv")  # parse the date column and tell pandas column 1 is a datetime
    pd.to_datetime(df.Date, dayfirst=True)
    return df


def get_bitcoin_prices():
    df = get_bitcoin_data()
    bitcoin_prices = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"}).set_index(df['Date'])
    return bitcoin_prices
