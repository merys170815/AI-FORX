import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================
API_KEY = "TU_API_KEY"
API_SECRET = "TU_API_SECRET"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 5000
CHECK_INTERVAL = 5 * 60  # 5 minutos

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
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit='ms')
    df.set_index("Open_time", inplace=True)

    # Evitar SettingWithCopyWarning
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def add_features_full(df, feature_cols):
    """Genera todas las features necesarias para tu modelo."""
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

    # Retardos de precios y volumen
    for lag in range(1, 6):
        df[f'ret_{lag}'] = df['ret'].shift(lag)
        df[f'vol_{lag}'] = df['Volume'].shift(lag)
        df[f'close_{lag}'] = df['Close'].shift(lag)

    # Asegurar todas las columnas del modelo
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols]
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def predict_signal(model, df):
    """Devuelve la probabilidad y señal en español."""
    X = df.iloc[-1:].copy()
    X.ffill(inplace=True)
    X.bfill(inplace=True)
    prob = model.predict_proba(X)[0][1]

    if prob >= UP_THRESHOLD:
        signal = "🟢 Comprar"
    elif prob <= DOWN_THRESHOLD:
        signal = "🔴 Vender"
    else:
        signal = "⚪ Mantener"

    return prob, signal

# ==============================
# 🚀 INICIO
# ==============================
try:
    client = Client(API_KEY, API_SECRET)
    client.futures_ping()
    print("✅ Conexión a Binance Futures OK")
except Exception as e:
    print("❌ Error al conectar a Binance:", e)
    exit(1)

print("🧠 Cargando modelo entrenado y features...")
model = joblib.load("binance_ai_lgbm_optuna_full.pkl")
feature_cols = model.feature_name_  # columnas exactas que usaste en el entrenamiento
print("✅ Modelo y features cargados correctamente.")
print(f"⏳ Monitoreando en tiempo real ({INTERVAL}) cada {CHECK_INTERVAL // 60} minutos...\n")

last_checked = {}

while True:
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] 🔄 Actualizando señales...")

        for symbol in SYMBOLS:
            df = download_klines_safe(symbol, INTERVAL, HISTORICAL_LIMIT)
            if df.empty:
                print(f"⚠️ {symbol}: no hay datos descargados, se omite.")
                continue

            last_close = df.index[-1]
            if symbol in last_checked and last_close == last_checked[symbol]:
                continue

            df_features = add_features_full(df, feature_cols)
            prob, signal = predict_signal(model, df_features)
            print(f"   {symbol:<10} → {signal} | Prob subida: {prob:.3f}")
            last_checked[symbol] = last_close

        time.sleep(CHECK_INTERVAL)

    except BinanceAPIException as e:
        print(f"⚠️ Error API Binance: {e}")
        time.sleep(10)
    except Exception as e:
        print(f"❌ Error general: {e}")
        time.sleep(10)
