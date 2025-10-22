import os
import logging
import joblib
import numpy as np
import pandas as pd
import ta
from binance.client import Client
from datetime import datetime

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"

# Umbrales iniciales
UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.55
DIFF_MARGIN = 0.05
ADX_THRESHOLD = 15
ATR_RATIO_THRESHOLD = 0.001

# SL/TP múltiplos ATR
SL_MULT = 1.0
TP_MULT = 2.0

# Suavizado de probabilidades
PROB_SMOOTH_WINDOW = 3

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------- INICIALIZAR CLIENTE ----------------
def init_client(api_key, api_secret):
    client = Client(api_key, api_secret)
    try:
        client.futures_ping()
        logging.info("✅ Conectado correctamente a Binance Futures.")
    except Exception as e:
        logging.error(f"Error conectando a Binance: {e}")
        return None
    return client


client = init_client(API_KEY, API_SECRET)
if client is None:
    exit(1)


# ---------------- FUNCIONES AUXILIARES ----------------
def download_klines(sym):
    try:
        kl = client.futures_klines(symbol=sym, interval=INTERVAL, limit=HISTORICAL_LIMIT)
        df = pd.DataFrame(kl, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
            "Quote_asset_volume", "Number_of_trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = df[c].astype(float)
        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
        df.set_index("Open_time", inplace=True)
        return df.ffill().bfill()
    except Exception as e:
        logging.error(f"Error descargando {sym}: {e}")
        return pd.DataFrame()


def compute_indicators(df):
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["ema_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    return df.ffill().bfill()


def smooth_series(arr, window=PROB_SMOOTH_WINDOW):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).mean().to_numpy()


# ---------------- BACKTEST ----------------
def backtest_symbol(df, model, feature_cols, initial_capital=10000):
    df_feat = compute_indicators(df)
    X = df_feat.reindex(columns=feature_cols).astype(float).fillna(0.0)

    try:
        preds = model.predict_proba(X)
    except Exception as e:
        logging.error(f"Error prediciendo: {e}")
        return 0, 0.0, 0.0, initial_capital, []

    classes = list(getattr(model, "classes_", []))
    prob_up = smooth_series(preds[:, classes.index(1)] if 1 in classes else preds[:, -1])
    prob_down = smooth_series(preds[:, classes.index(-1)] if -1 in classes else preds[:, 0])

    capital = initial_capital
    trade_log = []

    for i in range(1, len(df_feat) - 1):
        p_up = prob_up[i]
        p_down = prob_down[i]
        last_price = df_feat["Close"].iloc[i]
        last_atr = df_feat["atr"].iloc[i]
        last_adx = df_feat["adx"].iloc[i]
        atr_ratio = last_atr / (last_price + 1e-9)

        # Filtros de ruido
        if last_adx < ADX_THRESHOLD or atr_ratio < ATR_RATIO_THRESHOLD:
            continue

        signal = None
        if (p_up > UP_THRESHOLD) and (p_up - p_down >= DIFF_MARGIN):
            signal = "LONG"
        elif (p_down > DOWN_THRESHOLD) and (p_down - p_up >= DIFF_MARGIN):
            signal = "SHORT"

        if signal is None:
            continue

        # Calcular SL/TP
        if signal == "LONG":
            sl = last_price - SL_MULT * last_atr
            tp = last_price + TP_MULT * last_atr
        else:
            sl = last_price + SL_MULT * last_atr
            tp = last_price - TP_MULT * last_atr

        # Buscar cierre futuro hasta que SL/TP toque
        exit_price = None
        for j in range(i + 1, len(df_feat)):
            high, low, close = df_feat["High"].iloc[j], df_feat["Low"].iloc[j], df_feat["Close"].iloc[j]
            if signal == "LONG":
                if low <= sl:
                    exit_price = sl
                    break
                elif high >= tp:
                    exit_price = tp
                    break
            else:
                if high >= sl:
                    exit_price = sl
                    break
                elif low <= tp:
                    exit_price = tp
                    break
        if exit_price is None:
            exit_price = df_feat["Close"].iloc[-1]

        # Calcular ganancia %
        ret = (exit_price - last_price) / last_price if signal == "LONG" else (last_price - exit_price) / last_price
        capital *= (1 + ret)
        trade_log.append({
            "entry_idx": i,
            "signal": signal,
            "entry_price": last_price,
            "exit_price": exit_price,
            "return": ret,
            "capital": capital,
            "strength": abs(p_up - p_down)
        })

    if trade_log:
        total_trades = len(trade_log)
        wins = sum(1 for t in trade_log if t["return"] > 0)
        win_rate = wins / total_trades * 100
        gain = capital - initial_capital
    else:
        total_trades = 0
        win_rate = 0.0
        gain = 0.0

    return total_trades, win_rate, gain, capital, trade_log


# ---------------- MAIN ----------------
logging.info("Cargando modelo IA...")
model_data = joblib.load(MODEL_FILE)
if isinstance(model_data, dict):
    model = model_data["model"]
    feature_cols = model_data.get("features", list(getattr(model, "feature_name_", [])))
else:
    model = model_data
    feature_cols = list(getattr(model, "feature_name_", []))

logging.info(f"✅ Modelo cargado correctamente. Features: {feature_cols}")

for sym in SYMBOLS:
    df = download_klines(sym)
    if df.empty:
        continue
    trades, win_rate, gain, capital, log_trades = backtest_symbol(df, model, feature_cols)
    logging.info(
        f"{sym} → Trades: {trades}, Acierto: {win_rate:.2f}%, Ganancia: {gain:.2f} USDT, Capital final: {capital:.2f} USDT")
