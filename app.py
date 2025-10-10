from flask import Flask, jsonify, render_template
from flask_cors import CORS
from joblib import load
import yfinance as yf
import ta
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL = load("fx_model_ai_pro.pkl")
TICKER = "EURUSD=X"

def build_latest_features(ticker=TICKER, period="5d", interval="1h"):
    # Descargar datos
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        return None

    # Normalizar nombres de columnas
    df.columns = [str(c).capitalize() for c in df.columns]

    # Usar 'Adj close' en lugar de 'Close'
    price_col = "Adj close"

    # Indicadores técnicos
    df["ema20"] = ta.trend.EMAIndicator(df[price_col], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df[price_col], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df[price_col], window=200).ema_indicator()
    df["rsi14"] = ta.momentum.RSIIndicator(df[price_col], window=14).rsi()
    macd = ta.trend.MACD(df[price_col])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["atr14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df[price_col], window=14).average_true_range()

    # Rendimientos y volatilidad
    df["ret_1"] = df[price_col].pct_change(1)
    df["ret_3"] = df[price_col].pct_change(3)
    df["vol_rolling"] = df["ret_1"].rolling(24).std()

    # Soportes y resistencias
    df["sr_high20"] = df["High"].rolling(20).max()
    df["sr_low20"] = df["Low"].rolling(20).min()

    # Lags
    for lag in (1,2,3):
        df[f"close_lag_{lag}"] = df[price_col].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi14"].shift(lag)

    df = df.dropna()
    if df.empty:
        return None

    return df.iloc[-1:]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/signal", methods=["GET"])
def api_signal():
    feats = build_latest_features()
    if feats is None:
        return jsonify({"error": "no data"}), 500

    feature_cols = [
        "ema20","ema50","ema200","rsi14","macd","macd_hist","atr14",
        "ret_1","ret_3","vol_rolling","sr_high20","sr_low20",
        "close_lag_1","close_lag_2","close_lag_3","rsi_lag_1","rsi_lag_2","rsi_lag_3"
    ]
    X = feats[feature_cols]
    p_up = float(MODEL.predict_proba(X)[:,1][0])

    if p_up > 0.65:
        signal = "LONG"
    elif p_up < 0.35:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return jsonify({
        "ticker": TICKER,
        "p_up": round(p_up, 4),
        "signal": signal
    })

if __name__ == "__main__":
    # Permitir acceso desde celular (opcional: host="0.0.0.0")
    app.run(host="0.0.0.0", port=5000, debug=True)





