from fastapi import FastAPI
from pydantic import BaseModel
from data import fetch_data
from model import prepare_data, train_xgboost, train_lstm
import numpy as np

class CryptoInput(BaseModel):
    pair: str = "BTCUSDT"
    interval: str = "1d"
    limit: int = 1000

app = FastAPI(title="Crypto Predictor API")

# Step 0: Fetch + train model on startup
print("Fetching data and training model...")
df = fetch_data(symbol="BTCUSDT", interval="15m", limit=500)
X_seq, y_seq, scaler = prepare_data(df)
xgb_model, xgb_acc = train_xgboost(X_seq, y_seq)
lstm_model, lstm_acc, X_train, X_test, y_train, y_test = train_lstm(X_seq, y_seq, epochs=10)
print("Model trained!")

# Step 1: Prediction endpoint
@app.post("/predict")
def predict_crypto(data: CryptoInput):
    new_df = fetch_data(data.pair, data.interval, data.limit)
    X_seq_new, _, _ = prepare_data(new_df)
    
    # XGBoost
    last_flat = X_seq_new[-1].reshape(1, -1)
    xgb_prob = xgb_model.predict_proba(last_flat)[0][1]

    # LSTM
    last_seq = X_seq_new[-1].reshape(1, X_seq_new.shape[1], X_seq_new.shape[2])
    lstm_prob = lstm_model.predict(last_seq)[0][0]

    # Ensemble
    final_prob = (xgb_prob + lstm_prob)/2
    direction = "UP" if final_prob > 0.5 else "DOWN"

    return {"pair": data.pair, "direction": direction, "confidence": round(float(final_prob),2)}
