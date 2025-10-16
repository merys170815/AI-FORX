from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os, joblib
import pandas as pd
import ta
from datetime import datetime, timezone
from binance.client import Client

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

def init_client_testnet(api_key, api_secret):
    c = Client(api_key, api_secret, testnet=True)
    c.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return c


def download_klines_safe(sym, interval, limit=HISTORICAL_LIMIT):
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

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)

# Inicializar cliente y modelo
client = init_client_testnet(API_KEY, API_SECRET)

try:
    client.futures_ping()
    print("✅ Conexión con Binance REAL OK")
except Exception as e:
    print("❌ No se pudo conectar:", e)
    raise SystemExit(1)

print("Cargando modelo IA multiclass...")
model_dict = joblib.load(MODEL_FILE)
model = model_dict["model"]
feature_cols = model_dict["features"]
print("Modelo cargado. Features:", len(feature_cols))

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
                "prob": round(prob_signal,3),
                "price": round(entry_price,2) if entry_price else None,
                "SL": round(stop_loss,2) if stop_loss else None,
                "TP": round(take_profit,2) if take_profit else None
            })

        except Exception as e:
            signals.append({"symbol": sym, "error": str(e)})

    return jsonify(signals)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
