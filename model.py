import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------
# Technical indicators
# ------------------------
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

# ------------------------
# Data preparation
# ------------------------
def prepare_data(df, seq_len=30):
    df["returns"] = df["close"].pct_change()
    df["RSI"] = rsi(df["close"])
    df["MACD"] = macd(df["close"])
    df = df.dropna()

    features = ["open","high","low","close","volume","RSI","MACD"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])

    # sequences
    X_seq, y = [], []
    for i in range(seq_len, len(df)):
        X_seq.append(X_scaled[i-seq_len:i])
        y.append(1 if df["close"].iloc[i] > df["close"].iloc[i-1] else 0)

    return np.array(X_seq), np.array(y), scaler

# ------------------------
# XGBoost
# ------------------------
def train_xgboost(X_seq, y_seq):
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_seq, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

# ------------------------
# LSTM
# ------------------------
def train_lstm(X_seq, y_seq, epochs=30):
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    return model, acc, X_train, X_test, y_train, y_test
