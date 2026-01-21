import requests
import pandas as pd

def fetch_data(symbol="BTCUSDT", interval="5m", limit=5000):
    """Fetch historical candlestick data from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    return df
