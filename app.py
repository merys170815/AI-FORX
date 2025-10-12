from flask import Flask, jsonify, render_template
from flask_cors import CORS
from joblib import load
import pandas as pd
import numpy as np
import time
from binance.client import Client

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================
API_KEY = "rwO7WxY0j60W6yBCnZJRzR9lkypVYmywogIQ4cpV0sHkeecaVp2ebQoRz5EZWvht"
API_SECRET = "zO8UrZDHbOOdxZmz1kVrPHebmtnyYLTymAzPsBalsoQsuJto77AZ9qd1UzH7RuN0"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 5000

UP_THRESHOLD = 0.75
DOWN_THRESHOLD = 0.05

# ==============================
# ⚡ INDICADORES
# ==============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ==============================
# ⚙️ FUNCIONES PRINCIPALES
# ==============================
def download_klines_safe(symbol, interval, total_limit=5000):
    all_klines = []
    remaining = total_limit
    last_ts = None
    attempts = 0
    while remaining > 0 and attempts < 50:
        batch = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if last_ts:
            params["endTime"] = last_ts - 1
        kl = client.futures_klines(**params)
        if not kl:
            break
        all_klines = kl + all_klines
        remaining -= len(kl)
        last_ts = int(kl[0][0])
        attempts += 1
        time.sleep(0.2)
    df = pd.DataFrame(all_klines, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Trades', 'Taker_buy_base',
        'Taker_buy_quote', 'Ignore'
    ])
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = df[col].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit='ms')
    df.set_index("Open_time", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def add_features_full(df, feature_cols):
    df = df.copy()
    df['ret'] = df['Close'].pct_change()
    df['ema_20'] = df['Close'].ewm(span=20).mean()
    df['ema_50'] = df['Close'].ewm(span=50).mean()
    df['ema_100'] = df['Close'].ewm(span=100).mean()
    df['ema_diff_20_50'] = df['ema_20'] - df['ema_50']
    df['ema_diff_50_100'] = df['ema_50'] - df['ema_100']
    df['rsi'] = compute_rsi(df['Close'])
    df['macd'], df['signal'] = compute_macd(df['Close'])
    df['atr'] = compute_atr(df)
    df['vol_diff'] = df['Volume'] - df['Volume'].shift(1)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    for lag in range(1, 6):
        df[f'ret_{lag}'] = df['ret'].shift(lag)
        df[f'vol_{lag}'] = df['Volume'].shift(lag)
        df[f'close_{lag}'] = df['Close'].shift(lag)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def predict_signal(model, df):
    X = df.iloc[-1:].copy()
    prob = model.predict_proba(X)[0][1]
    if prob >= UP_THRESHOLD:
        signal = "🟢 Comprar"
    elif prob <= DOWN_THRESHOLD:
        signal = "🔴 Vender"
    else:
        signal = "⚪ Mantener"
    return prob, signal

# ==============================
# 🚀 INICIO DE APP WEB
# ==============================
app = Flask(__name__)
CORS(app)

try:
    client = Client(API_KEY, API_SECRET)
    client.futures_ping()
    print("✅ Conexión a Binance Futures OK")
except Exception as e:
    print("❌ Error al conectar a Binance:", e)
    exit(1)

print("🧠 Cargando modelo y features...")
model = load("binance_ai_lgbm_optuna_full.pkl")
feature_cols = model.feature_name_
print("✅ Modelo cargado correctamente.")

@app.route("/")
def home():
    return render_template("index.html")  # tu HTML para mostrar señales

@app.route("/api/signals")
def api_signals():
    results = []
    for symbol in SYMBOLS:
        df = download_klines_safe(symbol, INTERVAL, HISTORICAL_LIMIT)
        df_features = add_features_full(df, feature_cols)
        prob, signal = predict_signal(model, df_features)
        results.append({
            "symbol": symbol,
            "prob_up": round(prob, 3),
            "signal": signal
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
