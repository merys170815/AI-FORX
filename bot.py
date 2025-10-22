# ===============================
# 📈 IA TRADING ASSISTANT PRO — HÍBRIDO + AUTO RELOAD + CEREBRO + NOTICIAS + ENTRADAS FIB/ATR
# ===============================
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import ta
import threading
import time
import requests
from flask import Flask, jsonify, render_template, request
from binance.client import Client
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"
THRESHOLD_STATE_FILE = "thresholds_state.json"

# 🔧 UMBRALES INICIALES (IA)
UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.55
DIFF_MARGIN = 0.05

# 🔒 Cerebro estratégico
MIN_ATR_RATIO = 0.0005   # Volatilidad mínima (ATR/Precio)
MIN_ADX = 15             # Fuerza mínima de tendencia

# 🌀 Parámetros Fibonacci (híbrido)
FIB_LOOKBACK = 60          # Velas para detectar el último swing
FIB_ENTRY_LONG = 0.618     # Entrada sugerida long: retroceso 61.8%
FIB_ENTRY_SHORT = 0.382    # Entrada sugerida short: retroceso 38.2%
FIB_TP_EXT = 1.618         # TP en extensión 161.8% del rango
SL_BUFFER_ATR_MULT = 0.2   # Colchón ATR para SL
MIN_RANGE_RATIO = 0.003    # Rango mínimo (high-low)/price para usar FIB

# ⚠️ Noticias — configuración
TE_API_KEY = os.getenv("TE_API_KEY") or ""   # opcional
NEWS_LOOKAHEAD_MIN = 30

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- CLIENTE BINANCE ----------------
def init_client(api_key, api_secret, max_retries=5, backoff=2):
    for attempt in range(max_retries):
        try:
            client = Client(api_key, api_secret)
            client.futures_ping()
            logging.info("✅ Conectado correctamente a Binance Futures.")
            return client
        except Exception as e:
            logging.warning(f"Intento {attempt + 1} fallido conectando a Binance: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff ** attempt + np.random.uniform(0, 1)
                logging.info(f"Reintentando en {sleep_time:.2f} segundos...")
                time.sleep(sleep_time)
            else:
                logging.error("⚠️ No se pudo conectar a Binance después de varios intentos.")
                return None

# ---------------- 📢 NOTICIAS MACRO — DETALLE ----------------
def hay_noticia_importante_proxima():
    """
    Si hay una noticia de alto impacto en los próximos NEWS_LOOKAHEAD_MIN minutos,
    devuelve {"event","country","time"}. Si no hay, devuelve None.
    """
    try:
        url = "https://api.tradingeconomics.com/calendar"
        params = {"importance": "3", "limit": 20}
        if TE_API_KEY:
            params["c"] = TE_API_KEY

        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None

        data = r.json()
        now = datetime.utcnow()
        limit_time = now + timedelta(minutes=NEWS_LOOKAHEAD_MIN)

        for evento in data:
            event_time_str = evento.get("Date") or evento.get("DateSpan")
            if not event_time_str:
                continue
            try:
                event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                continue

            if now <= event_time <= limit_time:
                event_name = evento.get("Event", "Evento económico")
                country = evento.get("Country", "Desconocido")
                hora_str = event_time.strftime("%H:%M UTC")
                logging.warning(f"🚨 Noticia detectada: {event_name} ({hora_str}) [{country}]")
                return {"event": event_name, "country": country, "time": hora_str}
        return None
    except Exception as e:
        logging.warning(f"Error revisando noticias: {e}")
        return None

# ---------------- CARGA DE MODELO ----------------
logging.info("Cargando modelo IA...")
model = None
feature_cols = None
last_model_time = None

def load_model():
    global model, feature_cols, last_model_time
    try:
        mtime = os.path.getmtime(MODEL_FILE)
        if last_model_time is None or mtime != last_model_time:
            data = joblib.load(MODEL_FILE)
            if isinstance(data, dict):
                model = data.get("model")
                feature_cols = data.get("features")
            else:
                model = data
                feature_cols = list(getattr(model, "feature_name_", []) or [])
            last_model_time = mtime
            logging.info(f"♻️ Modelo cargado/actualizado correctamente ({MODEL_FILE})")
    except Exception as e:
        logging.exception(f"Fallo cargando modelo '{MODEL_FILE}': {e}")

load_model()

def auto_reload_model():
    while True:
        time.sleep(60)
        load_model()

threading.Thread(target=auto_reload_model, daemon=True).start()

if feature_cols is None:
    feature_cols = ["ema_9", "ema_26", "rsi", "atr", "bb_pct", "bb_width", "adx"]

# ---------------- AUXILIARES DATA ----------------
def download_klines_safe(sym):
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
        logging.error(f"Error descargando datos de {sym}: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["ema_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    return df.ffill().bfill()

# ---------------- CEREBRO (condiciones globales) ----------------
def condiciones_de_mercado_ok_df(symbol, df_feat):
    """
    Evalúa si vale la pena operar AHORA. Devuelve (ok: bool, reason: str, news: dict|None)
    """
    news = hay_noticia_importante_proxima()
    if news:
        return False, "Noticia de alto impacto próxima", news

    for col in ("Close", "atr", "adx"):
        if col not in df_feat.columns or df_feat[col].isna().iloc[-1]:
            return False, "Datos insuficientes", None

    price = float(df_feat["Close"].iloc[-1])
    atr = float(df_feat["atr"].iloc[-1])
    adx = float(df_feat["adx"].iloc[-1])

    atr_ratio = atr / (price + 1e-9)
    if atr_ratio < MIN_ATR_RATIO:
        return False, f"Volatilidad baja (ATR/Precio={atr_ratio:.5f} < {MIN_ATR_RATIO})", None

    if adx < MIN_ADX:
        return False, f"Tendencia débil (ADX={adx:.2f} < {MIN_ADX})", None

    return True, "OK", None

# ---------------- FIBONACCI / ATR (plan de trade) ----------------
def detect_swing_range(df_feat, lookback=FIB_LOOKBACK):
    """
    Detecta el rango (swing) reciente: retorna (low, high).
    Si hay pocos datos, devuelve None.
    """
    if len(df_feat) < lookback:
        return None, None
    window = df_feat.iloc[-lookback:]
    high = float(window["High"].max())
    low = float(window["Low"].min())
    return low, high

def build_fib_levels(low, high):
    """
    Calcula niveles de retroceso y extensión dado un rango low-high.
    Retorna dict con niveles clave.
    """
    rng = high - low
    levels = {
        "0.0": low,
        "38.2": high - rng * 0.382,
        "50.0": high - rng * 0.500,
        "61.8": high - rng * 0.618,
        "100": high,
        "161.8ext_up": high + rng * 0.618,
        "161.8ext_dn": low  - rng * 0.618,
        "range": rng,
    }
    return levels

def choose_trade_plan(signal, df_feat):
    """
    El cerebro ya decidió la dirección. Construimos un plan híbrido:
      - Si el rango relativo es suficiente y tendencia/volatilidad ok -> FIB
      - Si no -> ATR (fallback)
    Devuelve dict: {"plan","entry_suggest","SL","TP","risk_rr", "fib":{... opcional}}
    """
    price = float(df_feat["Close"].iloc[-1])
    atr = float(df_feat["atr"].iloc[-1])
    adx = float(df_feat["adx"].iloc[-1])
    ema20 = ta.trend.EMAIndicator(df_feat["Close"], window=20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(df_feat["Close"], window=50).ema_indicator().iloc[-1]

    # swing detect
    low, high = detect_swing_range(df_feat, FIB_LOOKBACK)
    if low is None or high is None or high <= low:
        return build_atr_plan(signal, price, atr)

    levels = build_fib_levels(low, high)
    rng_ratio = (levels["range"] / (price + 1e-9))

    # Condición para usar FIB
    use_fib = (rng_ratio >= MIN_RANGE_RATIO) and (adx >= MIN_ADX)

    if "COMPRA" in signal.upper() and use_fib and ema20 >= ema50:
        entry = float(levels["61.8"])  # pullback sano
        sl = float(low - atr * SL_BUFFER_ATR_MULT)  # por debajo del swing con colchón
        tp = float(levels["161.8ext_up"])  # extensión a favor
        plan = "FIB"
    elif "VENTA" in signal.upper() and use_fib and ema20 <= ema50:
        # Para shorts, medimos desde low->high; retroceso hacia 38.2% (rebote) para vender mejor
        entry = float(levels["38.2"])
        sl = float(high + atr * SL_BUFFER_ATR_MULT)
        tp = float(levels["161.8ext_dn"])
        plan = "FIB"
    else:
        return build_atr_plan(signal, price, atr)

    rr = compute_rr(entry, sl, tp)
    return {
        "plan": plan,
        "entry_suggest": round(entry, 4),
        "SL": round(sl, 4),
        "TP": round(tp, 4),
        "risk_rr": rr,
        "fib": {
            "low": round(low, 4),
            "high": round(high, 4),
            "r382": round(levels["38.2"], 4),
            "r50": round(levels["50.0"], 4),
            "r618": round(levels["61.8"], 4),
            "ext_up_1618": round(levels["161.8ext_up"], 4),
            "ext_dn_1618": round(levels["161.8ext_dn"], 4),
        }
    }

def build_atr_plan(signal, price, atr):
    """
    Respaldo ATR: entrada a mercado, SL = 1*ATR, TP = 1.5*ATR.
    """
    if "COMPRA" in signal.upper():
        entry = price
        sl = price - atr
        tp = price + atr * 1.5
    elif "VENTA" in signal.upper():
        entry = price
        sl = price + atr
        tp = price - atr * 1.5
    else:
        return {"plan": "NONE", "entry_suggest": None, "SL": None, "TP": None, "risk_rr": None}

    rr = compute_rr(entry, sl, tp)
    return {
        "plan": "ATR",
        "entry_suggest": round(entry, 4),
        "SL": round(sl, 4),
        "TP": round(tp, 4),
        "risk_rr": rr
    }

def compute_rr(entry, sl, tp):
    """
    Calcula el R:R (reward:risk). Devuelve string formato "x.xx".
    """
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 1e-12:
            return None
        return round(reward / risk, 2)
    except Exception:
        return None

# ---------------- NARRATIVA ----------------
def market_context_narrative(symbol, df, signal, prob_up, prob_down):
    try:
        lookback = 50
        price = df["Close"].iloc[-1]
        high_zone = max(df["High"].iloc[-lookback:])
        low_zone = min(df["Low"].iloc[-lookback:])
        atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range().iloc[-1]

        atr_distance_high = abs(price - high_zone) / (atr + 1e-9)
        atr_distance_low = abs(price - low_zone) / (atr + 1e-9)

        ema20 = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator().iloc[-1]
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx().iloc[-1]
        bb_width = (df["High"].iloc[-20:].max() - df["Low"].iloc[-20:].min()) / price

        regime = "lateral"
        if ema20 > ema50 and adx > 20 and bb_width > 0.015:
            regime = "alcista"
        elif ema20 < ema50 and adx > 20 and bb_width > 0.015:
            regime = "bajista"

        narrative = f"📊 *{symbol}* — Precio actual: {price:.2f}\n"
        narrative += f"📈 Prob ↑: {prob_up:.2f}% | 📉 Prob ↓: {prob_down:.2f}%\n"

        if atr_distance_high <= 1.0:
            narrative += f"🚀 Cerca de resistencia en {high_zone:.2f}. "
        elif atr_distance_low <= 1.0:
            narrative += f"🛡️ Cerca de soporte en {low_zone:.2f}. "
        else:
            narrative += "⚖️ Sin S/R relevantes inmediatos. "

        narrative += f"\n📊 Régimen actual: **{regime}**. "

        if "COMPRA" in signal.upper():
            narrative += "✅ Compra confirmada." if "CONFIRMADA" in signal.upper() else "⚠️ Oportunidad de compra sin confirmar."
        elif "VENTA" in signal.upper():
            narrative += "🔻 Venta confirmada." if "CONFIRMADA" in signal.upper() else "⚠️ Oportunidad de venta sin confirmar."
        else:
            narrative += "⏳ Sin señal clara."

        return narrative
    except Exception as e:
        logging.warning(f"Error narrativa {symbol}: {e}")
        return f"⚠️ No se pudo generar narrativa para {symbol}."

# ---------------- RUTAS ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask_symbol():
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").upper().strip()
        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400

        df = download_klines_safe(symbol)
        if df.empty:
            return jsonify({"error": f"No hay datos para {symbol}"}), 404

        df_feat = compute_indicators(df)
        X_all = df_feat.reindex(columns=feature_cols).astype(float).fillna(0.0)

        ok, why, news = condiciones_de_mercado_ok_df(symbol, df_feat)
        if not ok:
            price_block = round(float(df_feat["Close"].iloc[-1]), 2)
            payload = {
                "symbol": symbol,
                "signal": "ESPERAR 🚨" if news else "ESPERAR 🧠",
                "message": "Noticia de alto impacto próxima — Evitar operar." if news else f"Condiciones no ideales: {why}",
                "price": price_block,
                "prob_up": None,
                "prob_down": None,
                "entry_suggest": None,
                "SL": None,
                "TP": None,
                "risk_rr": None,
                "plan": "NONE",
                "narrative": f"🧠 Filtro estratégico activo: {why}"
            }
            if news:
                payload["news_event"] = news
            return jsonify(payload)

        # IA -> probabilidades
        preds_all = model.predict_proba(X_all)
        classes = list(getattr(model, "classes_", []))
        idx_up = classes.index(1) if 1 in classes else -1
        idx_down = classes.index(-1) if -1 in classes else 0
        prob_up = float(preds_all[-1, idx_up]) * 100
        prob_down = float(preds_all[-1, idx_down]) * 100

        # Señal IA (dirección)
        signal = "ESPERAR"
        if (prob_up/100 > UP_THRESHOLD) and (prob_up/100 - prob_down/100 >= DIFF_MARGIN):
            signal = "COMPRAR"
        elif (prob_down/100 > DOWN_THRESHOLD) and (prob_down/100 - prob_up/100 >= DIFF_MARGIN):
            signal = "VENDER"

        # Confirmación técnica
        adx_val = df_feat["adx"].iloc[-1]
        ema20 = ta.trend.EMAIndicator(df_feat["Close"], window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(df_feat["Close"], window=50).ema_indicator().iloc[-1]
        confirmation = (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50) or \
                       (signal == "VENDER" and adx_val > 20 and ema20 < ema50)

        if signal == "COMPRAR":
            signal = "COMPRA CONFIRMADA ✅" if confirmation else "COMPRA POTENCIAL ⚠️"
        elif signal == "VENDER":
            signal = "VENTA CONFIRMADA ✅" if confirmation else "VENTA POTENCIAL ⚠️"

        # 🧮 Plan de trade (FIB/ATR) híbrido
        price = float(df_feat["Close"].iloc[-1])
        atr_value = float(df_feat["atr"].iloc[-1])
        trade_plan = choose_trade_plan(signal, df_feat)

        narrativa = market_context_narrative(symbol, df_feat, signal, prob_up, prob_down)

        resp = {
            "symbol": symbol,
            "signal": signal,
            "prob_up": round(prob_up, 2),
            "prob_down": round(prob_down, 2),
            "price": round(price, 2),
            "entry_suggest": trade_plan.get("entry_suggest"),
            "SL": trade_plan.get("SL"),
            "TP": trade_plan.get("TP"),
            "risk_rr": trade_plan.get("risk_rr"),
            "plan": trade_plan.get("plan"),
            "narrative": narrativa
        }
        if trade_plan.get("fib"):
            resp["fib"] = trade_plan["fib"]
        return jsonify(resp)

    except Exception as e:
        logging.exception("Error en /api/ask")
        return jsonify({"error": str(e)}), 500

@app.route("/api/signals")
def api_signals():
    signals_list = []
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500

        for sym in SYMBOLS:
            df = download_klines_safe(sym)
            if df.empty:
                continue
            df_feat = compute_indicators(df)

            ok, why, news = condiciones_de_mercado_ok_df(sym, df_feat)
            if not ok:
                item = {
                    "symbol": sym,
                    "signal": "ESPERAR 🚨" if news else "ESPERAR 🧠",
                    "prob_up": None,
                    "prob_down": None,
                    "price": round(float(df_feat["Close"].iloc[-1]), 4) if "Close" in df_feat else None,
                    "entry_suggest": None,
                    "SL": None,
                    "TP": None,
                    "risk_rr": None,
                    "plan": "NONE",
                    "reason": "Noticia de alto impacto próxima" if news else why
                }
                if news:
                    item["news_event"] = news
                signals_list.append(item)
                continue

            X_all = df_feat.reindex(columns=feature_cols).astype(float).fillna(0.0)
            preds_all = model.predict_proba(X_all)
            classes = list(getattr(model, "classes_", []))
            idx_up = classes.index(1) if 1 in classes else -1
            idx_down = classes.index(-1) if -1 in classes else 0
            prob_up = float(preds_all[-1, idx_up])
            prob_down = float(preds_all[-1, idx_down])

            signal = "ESPERAR"
            if (prob_up > UP_THRESHOLD) and (prob_up - prob_down >= DIFF_MARGIN):
                signal = "COMPRAR"
            elif (prob_down > DOWN_THRESHOLD) and (prob_down - prob_up >= DIFF_MARGIN):
                signal = "VENTA"

            # Confirmación y plan de trade
            if signal in ("COMPRAR", "VENTA"):
                signal_out = signal + (" CONFIRMADA ✅" if (
                    (signal == "COMPRAR" and df_feat["adx"].iloc[-1] > 20 and
                     ta.trend.EMAIndicator(df_feat["Close"], window=20).ema_indicator().iloc[-1] >
                     ta.trend.EMAIndicator(df_feat["Close"], window=50).ema_indicator().iloc[-1])
                    or
                    (signal == "VENTA" and df_feat["adx"].iloc[-1] > 20 and
                     ta.trend.EMAIndicator(df_feat["Close"], window=20).ema_indicator().iloc[-1] <
                     ta.trend.EMAIndicator(df_feat["Close"], window=50).ema_indicator().iloc[-1])
                ) else " POTENCIAL ⚠️")
            else:
                signal_out = "ESPERAR"

            price = float(df_feat["Close"].iloc[-1])
            plan = choose_trade_plan(signal_out, df_feat)

            item = {
                "symbol": sym,
                "signal": signal_out,
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "price": round(price, 4),
                "entry_suggest": plan.get("entry_suggest"),
                "SL": plan.get("SL"),
                "TP": plan.get("TP"),
                "risk_rr": plan.get("risk_rr"),
                "plan": plan.get("plan")
            }
            if plan.get("fib"):
                item["fib"] = plan["fib"]
            signals_list.append(item)

        return jsonify(signals_list)

    except Exception as e:
        logging.exception("Error en /api/signals")
        return jsonify({"error": str(e)}), 500

# ---------------- MAIN ----------------
if __name__ == "__main__":
    client = init_client(API_KEY, API_SECRET)
    if client is None:
        logging.error("No se pudo conectar a Binance. Saliendo.")
        exit(1)
    app.run(host="0.0.0.0", port=5000, debug=False)
