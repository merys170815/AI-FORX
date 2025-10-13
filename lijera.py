# lijera_real_ichimoku_signals.py
# ✅ Modo REAL (solo señales, sin trading)
# IA + Ichimoku + RSI + ATR confirmaciones

import os, time, math, csv
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY")       # asegúrate de tener tus claves configuradas
API_SECRET = os.getenv("API_SECRET")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
CHECK_INTERVAL = 5 * 60  # 5 minutos

UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.45

MODEL_FILE = "binance_ai_lgbm_optuna_full.pkl"
TRADES_LOG = "trades_log.csv"

# ---------------- FUNCIONES ----------------
def ichimoku(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
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
    return ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()

def init_client_real(api_key, api_secret):
    c = Client(api_key, api_secret)
    try:
        c.FUTURES_URL = "https://fapi.binance.com/fapi"
    except:
        pass
    return c

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

def log_trade(entry: dict):
    header = ["timestamp","symbol","signal","prob","price","extra"]
    write_header = not os.path.exists(TRADES_LOG)
    with open(TRADES_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            entry.get("timestamp"), entry.get("symbol"), entry.get("signal"),
            round(entry.get("prob",0),4), entry.get("price"), entry.get("extra","")
        ])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    client = init_client_real(API_KEY, API_SECRET)
    try:
        client.futures_ping()
        print("✅ Conexión con Binance REAL OK")
    except Exception as e:
        print("❌ No se pudo conectar:", e)
        raise SystemExit(1)

    print("Cargando modelo IA...")
    model = joblib.load(MODEL_FILE)
    feature_cols = model.feature_name_ if hasattr(model, "feature_name_") else model["features"]
    print("Modelo cargado. Features:", len(feature_cols))

    last_checked = {}

    while True:
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] Actualizando señales...")

            for sym in SYMBOLS:
                df = download_klines_safe(sym, INTERVAL, HISTORICAL_LIMIT)
                if df.empty:
                    print("Sin datos", sym)
                    continue
                last_close = df.index[-1]
                if sym in last_checked and last_checked[sym] == last_close:
                    continue

                # --- Features ---
                df_feat = df.copy()
                tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df_feat)
                df_feat["tenkan"] = tenkan
                df_feat["kijun"] = kijun
                df_feat["senkou_a"] = senkou_a
                df_feat["senkou_b"] = senkou_b
                df_feat["chikou"] = chikou
                df_feat["rsi"] = compute_rsi(df_feat['Close'])
                df_feat["atr"] = compute_atr(df_feat)
                for c in feature_cols:
                    if c not in df_feat.columns:
                        df_feat[c] = 0.0

                X_row = df_feat[feature_cols].ffill().bfill()
                prob = float(model.predict_proba(X_row.iloc[[-1]])[0][1])

                # --- Confirmaciones técnicas ---
                confirmations = 0
                price = df["Close"].iloc[-1]
                if price > max(senkou_a.iloc[-1], senkou_b.iloc[-1]): confirmations += 1
                if tenkan.iloc[-1] > kijun.iloc[-1]: confirmations += 1
                if chikou.iloc[-1] > price: confirmations += 1
                rsi = df_feat["rsi"].iloc[-1]
                if 40 <= rsi <= 70: confirmations += 1
                atr = df_feat["atr"].iloc[-1] if not pd.isna(df_feat["atr"].iloc[-1]) else 0.0
                if atr > 0 and (atr / price) < 0.02: confirmations += 1

                # --- Señal IA ---
                ia_signal = "HOLD"
                if prob >= UP_THRESHOLD:
                    ia_signal = "BUY"
                elif prob <= DOWN_THRESHOLD:
                    ia_signal = "SELL"

                print(f"{sym}: prob={prob:.3f} ia={ia_signal} conf={confirmations} price={price:.2f}")

                if ia_signal == "BUY" and confirmations >= 2:
                    print(f"📈 Señal de COMPRA detectada en {sym} (prob={prob:.3f}, conf={confirmations})")
                    log_trade({
                        "timestamp": now, "symbol": sym, "signal": "BUY",
                        "prob": prob, "price": price, "extra": f"conf={confirmations}"
                    })
                elif ia_signal == "SELL" and confirmations >= 2:
                    print(f"📉 Señal de VENTA detectada en {sym} (prob={prob:.3f}, conf={confirmations})")
                    log_trade({
                        "timestamp": now, "symbol": sym, "signal": "SELL",
                        "prob": prob, "price": price, "extra": f"conf={confirmations}"
                    })

                last_checked[sym] = last_close

            time.sleep(CHECK_INTERVAL)

        except BinanceAPIException as b:
            print("BinanceAPIException:", b)
            time.sleep(10)
        except Exception as e:
            print("Error general:", e)
            time.sleep(10)
