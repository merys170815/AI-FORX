# app.py
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from joblib import load
import yfinance as yf
import ta
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar modelo entrenado
MODEL = load("fx_model_ai_pro.pkl")
TICKER = "EURUSD=X"
P_UP_THRESHOLD = 0.65
P_DOWN_THRESHOLD = 0.35

def build_latest_features_safe(ticker=TICKER, period="20d", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval)
    print("Columnas descargadas (raw):", df.columns.tolist())

    if df.empty:
        return None

    # Aplanar MultiIndex de columnas si lo hay
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    print("Columnas aplanadas:", df.columns.tolist())

    # Detectar columna de precio
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    elif "Open" in df.columns:
        price_col = "Open"
    else:
        print("No hay columna de precio válida")
        return None

    print("Columna de precio usada:", price_col)

    # Indicadores técnicos (ajustamos ventanas para period corto)
    df["ema20"] = ta.trend.EMAIndicator(df[price_col], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df[price_col], window=50).ema_indicator()
    # Cambiamos ema200 a 50 si hay pocos datos
    win_ema200 = min(200, len(df))
    df["ema200"] = ta.trend.EMAIndicator(df[price_col], window=win_ema200).ema_indicator()
    df["rsi14"] = ta.momentum.RSIIndicator(df[price_col], window=14).rsi()
    macd = ta.trend.MACD(df[price_col])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    win_atr14 = min(14, len(df))
    df["atr14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df[price_col], window=win_atr14).average_true_range()

    # Rendimientos y volatilidad
    df["ret_1"] = df[price_col].pct_change(1)
    df["ret_3"] = df[price_col].pct_change(3)
    df["vol_rolling"] = df["ret_1"].rolling(24).std()

    # Soportes y resistencias
    df["sr_high20"] = df["High"].rolling(20).max()
    df["sr_low20"] = df["Low"].rolling(20).min()

    # Lags
    for lag in (1, 2, 3):
        df[f"close_lag_{lag}"] = df[price_col].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi14"].shift(lag)

    df = df.dropna()
    if df.empty:
        print("DataFrame vacío después de calcular indicadores")
        return None

    return df.iloc[-1:]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/signal", methods=["GET"])
def api_signal():
    feats = build_latest_features_safe()
    if feats is None:
        return jsonify({"error": "No hay datos recientes o faltan columnas de precio"}), 500

    feature_cols = [
        "ema20","ema50","ema200","rsi14","macd","macd_hist","atr14",
        "ret_1","ret_3","vol_rolling","sr_high20","sr_low20",
        "close_lag_1","close_lag_2","close_lag_3","rsi_lag_1","rsi_lag_2","rsi_lag_3"
    ]

    # Verificar columnas antes de predecir
    X = feats[feature_cols]
    print("Columnas para el modelo:", X.columns.tolist())
    print("Valores NaN en X:", X.isna().sum())

    if X.isna().any().any():
        return jsonify({"error": "Hay NaN en las features, no se puede predecir"}), 500

    p_up = float(MODEL.predict_proba(X)[:,1][0])

    if p_up > P_UP_THRESHOLD:
        signal = "LONG"
    elif p_up < P_DOWN_THRESHOLD:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return jsonify({
        "ticker": TICKER,
        "p_up": round(p_up, 4),
        "signal": signal
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)







