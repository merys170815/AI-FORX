from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib, time, os
from datetime import datetime, timezone
from binance.client import Client
import ta

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500

UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.45

MODEL_FILE = "binance_ai_lgbm_optuna_full.pkl"

# ==============================
# ⚙️ FUNCIONES DE INDICADORES
# ==============================
def ichimoku(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    ich = ta.trend.IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
    tenkan = ich.ichimoku_conversion_line()
    kijun = ich.ichimoku_base_line()
    senkou_a = ich.ichimoku_a()
    senkou_b = ich.ichimoku_b()
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou

def compute_rsi(close, period=14):
    return ta.momentum.RSIIndicator(close, window=period).rsi()

def compute_atr(df, window=14):
    return ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=window).average_true_range()

def download_klines_safe(sym, interval, limit=1500):
    df = pd.DataFrame(client.futures_klines(symbol=sym, interval=interval, limit=limit),
                      columns=["Open_time","Open","High","Low","Close","Volume","Close_time",
                               "Quote_asset_volume","Number_of_trades","Taker_buy_base","Taker_buy_quote","Ignore"])
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = df[c].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    df.set_index("Open_time", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ==============================
# 🚀 APP WEB FLASK
# ==============================
app = Flask(__name__)
CORS(app)

# Conectar a Binance
try:
    client = Client(API_KEY, API_SECRET)
    client.futures_ping()
    print("✅ Conexión con Binance REAL OK")
except Exception as e:
    print("❌ No se pudo conectar:", e)
    exit(1)

# Cargar modelo
print("🧠 Cargando modelo IA...")
model = joblib.load(MODEL_FILE)
feature_cols = model.feature_name_ if hasattr(model, "feature_name_") else model["features"]
print("✅ Modelo cargado correctamente.")

@app.route("/")
def home():
    return render_template("index.html")  # Puedes mostrar las señales aquí

@app.route("/api/signals")
def api_signals():
    signals = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for sym in SYMBOLS:
        try:
            df = download_klines_safe(sym, INTERVAL, HISTORICAL_LIMIT)
            tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df)
            df["tenkan"] = tenkan
            df["kijun"] = kijun
            df["senkou_a"] = senkou_a
            df["senkou_b"] = senkou_b
            df["chikou"] = chikou
            df["rsi"] = compute_rsi(df["Close"])
            df["atr"] = compute_atr(df)

            for c in feature_cols:
                if c not in df.columns:
                    df[c] = 0.0

            X = df[feature_cols].ffill().bfill()
            prob = float(model.predict_proba(X.iloc[[-1]])[0][1])
            price = df["Close"].iloc[-1]

            confirmations = 0
            if price > max(senkou_a.iloc[-1], senkou_b.iloc[-1]): confirmations += 1
            if tenkan.iloc[-1] > kijun.iloc[-1]: confirmations += 1
            if chikou.iloc[-1] > price: confirmations += 1
            rsi = df["rsi"].iloc[-1]
            if 40 <= rsi <= 70: confirmations += 1
            atr = df["atr"].iloc[-1] if not pd.isna(df["atr"].iloc[-1]) else 0.0
            if atr > 0 and (atr / price) < 0.02: confirmations += 1

            signal = "⚪ HOLD"
            if prob >= UP_THRESHOLD and confirmations >= 2:
                signal = "🟢 BUY"
            elif prob <= DOWN_THRESHOLD and confirmations >= 2:
                signal = "🔴 SELL"

            signals.append({
                "timestamp": now,
                "symbol": sym,
                "signal": signal,
                "prob": round(prob, 3),
                "confirmations": confirmations,
                "price": round(price, 2)
            })
        except Exception as e:
            signals.append({
                "symbol": sym,
                "error": str(e)
            })
            continue

    return jsonify(signals)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
