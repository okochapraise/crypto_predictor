from data import fetch_data
from model import prepare_data, train_xgboost, train_lstm
import numpy as np

def classify_confidence(prob):
    """
    Classify the confidence score into signal categories.
    """
    if 0.00 <= prob < 0.40:
        return "SELL (Strong)"
    elif 0.40 <= prob < 0.45:
        return "SELL (Weak)"
    elif 0.45 <= prob <= 0.55:
        return "HOLD/Uncertain"
    elif 0.55 < prob <= 0.60:
        return "BUY (Weak)"
    elif 0.60 < prob <= 1.00:
        return "BUY (Strong)"
    else:
        return "UNKNOWN"

def main():
    print("Fetching data...")
    df = fetch_data(symbol="BTCUSDT", interval="15m", limit=5000)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, seq_len=30)
    
    print("Training XGBoost...")
    xgb_model, xgb_acc = train_xgboost(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    print("Training LSTM...")
    lstm_model, lstm_acc, X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_lstm(
        np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), epochs=30
    )
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
    
    print("\nPredicting next candle direction...")
    
    # XGBoost
    last_flat = X_test[-1].reshape(1, -1)
    xgb_prob = xgb_model.predict_proba(last_flat)[0][1]
    
    # LSTM
    last_seq = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
    lstm_prob = lstm_model.predict(last_seq)[0][0]
    
    # Ensemble
    final_prob = (xgb_prob + lstm_prob) / 2
    signal_category = classify_confidence(final_prob)
    
    print("\n📊 FINAL TRADE SIGNAL")
    print("--------------------")
    print(f"Signal & Strength : {signal_category}")
    print(f"Confidence        : {final_prob:.2f}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
