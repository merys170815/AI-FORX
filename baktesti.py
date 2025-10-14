# baktesti_professional_model_1y_conservative.py
import os, joblib, time
import pandas as pd
import numpy as np
from binance.client import Client
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
API_KEY = os.getenv("BINANCE_API_KEY") or "TU_API_KEY_TEMPORAL"
API_SECRET = os.getenv("BINANCE_API_SECRET") or "TU_API_SECRET_TEMPORAL"
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT"]
INTERVAL = "1h"
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.05
COMMISSION = 0.0004
SLIPPAGE = 0.0005
UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.55
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"

# ---------------- FUNCIONES ----------------
def ichimoku_levels(df):
    ich = ta.trend.IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    return ich.ichimoku_conversion_line(), ich.ichimoku_base_line(), ich.ichimoku_a(), ich.ichimoku_b()

def compute_rsi(close, period=14):
    return ta.momentum.RSIIndicator(close, window=period).rsi()

def compute_atr(df, window=14):
    return ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()

def download_historical_klines(symbol, interval, start_date, end_date):
    client = Client(API_KEY, API_SECRET)
    kl = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    while start_ts < end_ts:
        data = client.futures_klines(symbol=symbol, interval=interval, limit=1000, startTime=start_ts)
        if not data:
            break
        kl += data
        start_ts = data[-1][0] + 1
    df = pd.DataFrame(kl, columns=["Open_time","Open","High","Low","Close","Volume",
                                   "Close_time","Quote_asset_volume","Number_of_trades",
                                   "Taker_buy_base","Taker_buy_quote","Ignore"])
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = df[c].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    df.set_index("Open_time", inplace=True)
    return df[["Open","High","Low","Close","Volume"]]

# ---------------- CARGAR MODELO ----------------
model_dict = joblib.load(MODEL_FILE)
model = model_dict["model"]
feature_cols = model_dict["features"]
scaler = model_dict.get("scaler", None)
scaled_cols = getattr(scaler, "feature_names_in_", None)
print(f"✅ Modelo cargado. Features: {len(feature_cols)}")

# ---------------- BACKTEST ----------------
results = []
today = datetime.utcnow()
start_date = today - timedelta(days=365)

for sym in SYMBOLS:
    print(f"\n📊 Backtesting {sym} (último año)...")
    df = download_historical_klines(sym, INTERVAL, start_date, today)

    # Indicadores para SL/TP
    df["tenkan"], df["kijun"], df["senkou_a"], df["senkou_b"] = ichimoku_levels(df)
    df["rsi"] = compute_rsi(df["Close"])
    df["atr"] = compute_atr(df)

    # Preparar features
    X = pd.DataFrame({c: df[c] if c in df.columns else 0.0 for c in feature_cols})
    if scaler:
        X_scaled = X.copy()
        common_cols = [c for c in scaled_cols if c in X_scaled.columns]
        X_scaled[common_cols] = scaler.transform(X_scaled[common_cols])
    else:
        X_scaled = X

    probs_all = model.predict_proba(X_scaled)
    capital = INITIAL_CAPITAL
    equity_curve = []

    for i in range(len(df)-1):
        price = df["Close"].iloc[i]
        prob_down, prob_neutral, prob_up = probs_all[i]

        # Señales IA
        if prob_up >= UP_THRESHOLD:
            signal = "BUY"
            prob_signal = prob_up
        elif prob_down >= DOWN_THRESHOLD:
            signal = "SELL"
            prob_signal = prob_down
        else:
            signal = "HOLD"
            prob_signal = max(probs_all[i])

        entry_price = price
        atr = df["atr"].iloc[i] if not pd.isna(df["atr"].iloc[i]) else 0.0

        # Estrategia conservadora
        if signal=="BUY":
            entry_price = min(price, df["senkou_b"].iloc[i], df["kijun"].iloc[i])
            sl = entry_price - 1.8*atr
            tp = entry_price + 2.5*(entry_price - sl)
        elif signal=="SELL":
            entry_price = max(price, df["senkou_b"].iloc[i], df["kijun"].iloc[i])
            sl = entry_price + 1.8*atr
            tp = entry_price - 2.5*(sl - entry_price)
        else:
            sl = tp = None

        next_price = df["Close"].iloc[i+1]
        pnl = 0.0
        if signal=="BUY":
            if next_price <= sl: pnl = -POSITION_SIZE*capital*(entry_price-sl)/entry_price
            elif next_price >= tp: pnl = POSITION_SIZE*capital*(tp-entry_price)/entry_price
            else: pnl = POSITION_SIZE*capital*(next_price-entry_price)/entry_price
        elif signal=="SELL":
            if next_price >= sl: pnl = -POSITION_SIZE*capital*(sl-entry_price)/entry_price
            elif next_price <= tp: pnl = POSITION_SIZE*capital*(entry_price-tp)/entry_price
            else: pnl = POSITION_SIZE*capital*(entry_price-next_price)/entry_price

        pnl -= POSITION_SIZE*capital*(COMMISSION+SLIPPAGE)
        capital += pnl

        if signal in ["BUY","SELL"]:
            results.append({
                "timestamp": df.index[i],
                "symbol": sym,
                "signal": signal,
                "prob": prob_signal,
                "entry": entry_price,
                "SL": sl,
                "TP": tp,
                "pnl": pnl,
                "capital": capital
            })

        equity_curve.append(capital)

    df_equity = pd.DataFrame({"capital": equity_curve}, index=df.index[:len(equity_curve)])
    df_equity.to_csv(f"equity_curve_{sym}_conservative.csv")

    plt.figure(figsize=(10,5))
    plt.plot(df_equity.index, df_equity["capital"], label=f"Equity {sym}")
    plt.title(f"Equity Curve {sym} (1 año, conservadora)")
    plt.xlabel("Fecha")
    plt.ylabel("Capital")
    plt.legend()
    plt.savefig(f"equity_curve_{sym}_conservative.png")
    plt.close()

    print(f"Capital final {sym}: ${capital:.2f}")

# ---------------- REPORTE FINAL ----------------
df_res = pd.DataFrame(results)
df_res.to_csv("backtest_results_1y_professional_model_conservative.csv", index=False)
print("\n✅ Backtest anual (versión conservadora) completado.")
print("Archivo: backtest_results_1y_professional_model_conservative.csv")
