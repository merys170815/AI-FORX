from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os, joblib
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone
# ⚠️ Eliminamos import de Binance ya que no lo usaremos en modo simulado

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500

UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.55

MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"

# ---------------- FUNCIONES ----------------
def ichimoku(df):
    high, low, close = df['High'], df['Low'], df['Close']
    ich = ta.trend.IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
    return ich.ichimoku_conversion_line(), ich.ichimoku_base_line(), ich.ichimoku_a(), ich.ichimoku_b(), close.shift(-26)

def compute_rsi(close, period=14):
    return ta.momentum.RSIIndicator(close, window=period).rsi()

def compute_atr(df, window=14):
    return ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()

# ---------------- SIMULADOR DE MERCADO ----------------
def download_klines_safe(sym, interval, limit=HISTORICAL_LIMIT):
    """
    Simula datos OHLCV para un símbolo.
    Mantiene estructura idéntica a Binance Futures.
    """
    np.random.seed(abs(hash(sym)) % 2**32)
    base_price = {
        "BTCUSDT": 68000,
        "ETHUSDT": 2500,
        "BNBUSDT": 600,
        "SOLUSDT": 150,
        "ADAUSDT": 0.45,
        "DOGEUSDT": 0.12
    }.get(sym, 1000)

    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="H")
    prices = base_price + np.cumsum(np.random.randn(limit))  # ruido aleatorio

    df = pd.DataFrame({
        "Open": prices + np.random.randn(limit),
        "High": prices + abs(np.random.randn(limit)),
        "Low": prices - abs(np.random.randn(limit)),
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, size=limit),
    }, index=dates)

    df.index.name = "Open_time"
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)

print("✅ Modo SIMULADO activado (sin conexión a Binance)")
print("Cargando modelo IA multiclass...")

model_dict = joblib.load(MODEL_FILE)
model = model_dict["model"]
feature_cols = model_dict["features"]
print("✅ Modelo cargado. Features:", len(feature_cols))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/signals")
def api_signals():
    signals = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for sym in SYMBOLS:
        try:
            df = download_klines_safe(sym, INTERVAL, HISTORICAL_LIMIT)
            if df.empty:
                continue

            # --- Indicadores ---
            df_feat = df.copy()
            tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df_feat)
            df_feat["tenkan"] = tenkan
            df_feat["kijun"] = kijun
            df_feat["senkou_a"] = senkou_a
            df_feat["senkou_b"] = senkou_b
            df_feat["chikou"] = chikou
            df_feat["rsi"] = compute_rsi(df_feat['Close'])
            df_feat["atr"] = compute_atr(df_feat)

            # --- Preparar features para IA ---
            for c in feature_cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0.0
            X_row = df_feat[feature_cols].ffill().bfill()

            # --- Predicción IA ---
            probs = model.predict_proba(X_row.iloc[[-1]])[0]
            prob_down, prob_neutral, prob_up = probs

            if prob_up >= UP_THRESHOLD:
                ia_signal = "BUY"
                prob_signal = prob_up
            elif prob_down >= DOWN_THRESHOLD:
                ia_signal = "SELL"
                prob_signal = prob_down
            else:
                ia_signal = "HOLD"
                prob_signal = max(probs)

            # --- Niveles profesionales ---
            price = df["Close"].iloc[-1]
            atr = df_feat["atr"].iloc[-1] if not pd.isna(df_feat["atr"].iloc[-1]) else 0.0
            entry_price = stop_loss = take_profit = None

            if ia_signal == "BUY":
                entry_price = min(price, senkou_b.iloc[-1], kijun.iloc[-1])
                stop_loss = entry_price - 1.5 * atr
                take_profit = entry_price + 2 * (entry_price - stop_loss)
            elif ia_signal == "SELL":
                entry_price = max(price, senkou_b.iloc[-1], kijun.iloc[-1])
                stop_loss = entry_price + 1.5 * atr
                take_profit = entry_price - 2 * (stop_loss - entry_price)

            signals.append({
                "timestamp": now,
                "symbol": sym,
                "signal": ia_signal,
                "prob": round(prob_signal, 3),
                "price": round(entry_price, 2) if entry_price else None,
                "SL": round(stop_loss, 2) if stop_loss else None,
                "TP": round(take_profit, 2) if take_profit else None
            })

        except Exception as e:
            signals.append({"symbol": sym, "error": str(e)})

    return jsonify(signals)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
