# ===============================
# 📈 IA TRADING ASSISTANT PRO — HÍBRIDO + AUTO RELOAD + CEREBRO + NOTICIAS + ENTRADAS FIB/ATR + SCANNER
# ✅ FIXED: probabilidades, mapeo clases, JSON thresholds, NaN warmup, parseo fechas, EMAs cacheadas, protección FIB
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
from datetime import datetime, timedelta, timezone
from dateutil import parser  # ✅ parseo robusto de fechas

# ====== 👇 SHAP opcional (degrada elegante si no está instalado) ======
try:
    import shap  # pip install shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"
THRESHOLD_STATE_FILE = "thresholds_state.json"

# 🔧 UMBRALES INICIALES (IA) — siempre comparar en [0–1] internamente
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

# ====== 👇 globals para explicabilidad ======
explainer_shap = None   # se inicializa tras cargar el modelo
model_is_tree = False   # pista para elegir TreeExplainer vs Kernel/Linear si hiciera falta

# ===================================================
# 🧠 PASO 2 — Thresholds dinámicos (Aprendizaje continuo)
# ===================================================

MAX_THRESHOLD = 0.80
MIN_THRESHOLD = 0.40
THRESHOLD_STEP = 0.01

# ✅ JSON inicial por símbolo
if not os.path.exists(THRESHOLD_STATE_FILE):
    with open(THRESHOLD_STATE_FILE, "w") as f:
        json.dump({"history": {}}, f)

def guardar_resultado_trade(symbol, signal, prob, result):
    """
    Guarda el resultado de un trade por símbolo para ajustar umbrales dinámicos.
    result: True = ganó | False = perdió
    """
    try:
        with open(THRESHOLD_STATE_FILE, "r") as f:
            data = json.load(f)
        history_dict = data.get("history", {})
        history = history_dict.get(symbol, [])

        history.append({
            "signal": signal,
            "prob": float(prob),
            "win": bool(result),
            "timestamp": datetime.utcnow().isoformat()
        })
        history = history[-500:]  # últimos 500 por símbolo
        history_dict[symbol] = history
        data["history"] = history_dict

        with open(THRESHOLD_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.warning(f"⚠️ Error guardando resultado de trade: {e}")

def recalcular_thresholds():
    """
    Recalcula dinámicamente los umbrales en función del winrate histórico (global).
    (Si quieres, puedes cambiar a por símbolo leyendo history_dict[symbol])
    """
    global UP_THRESHOLD, DOWN_THRESHOLD
    try:
        with open(THRESHOLD_STATE_FILE, "r") as f:
            data = json.load(f)
        history_dict = data.get("history", {})

        all_trades = [h for sym_hist in history_dict.values() for h in sym_hist]
        if len(all_trades) < 20:
            return  # no hay suficiente histórico aún

        wins = sum(1 for h in all_trades if h.get("win"))
        total = len(all_trades)
        winrate = wins / total if total else 0.0

        if winrate < 0.5 and UP_THRESHOLD < MAX_THRESHOLD:
            UP_THRESHOLD += THRESHOLD_STEP
            DOWN_THRESHOLD += THRESHOLD_STEP
        elif winrate > 0.7 and UP_THRESHOLD > MIN_THRESHOLD:
            UP_THRESHOLD -= THRESHOLD_STEP
            DOWN_THRESHOLD -= THRESHOLD_STEP

        UP_THRESHOLD = max(MIN_THRESHOLD, min(MAX_THRESHOLD, UP_THRESHOLD))
        DOWN_THRESHOLD = max(MIN_THRESHOLD, min(MAX_THRESHOLD, DOWN_THRESHOLD))

        logging.info(f"📊 Thresholds actualizados → UP: {UP_THRESHOLD:.2f} | DOWN: {DOWN_THRESHOLD:.2f} | Winrate: {winrate:.2%}")
    except Exception as e:
        logging.warning(f"⚠️ Error recalculando thresholds: {e}")

def auto_adjust_thresholds():
    """Hilo en background que ajusta thresholds cada hora automáticamente."""
    while True:
        time.sleep(3600)
        recalcular_thresholds()

# Lanzar hilo de ajuste automático de umbrales
threading.Thread(target=auto_adjust_thresholds, daemon=True).start()

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
    ✅ Usa dateutil.parser.isoparse y normaliza a UTC.
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
        now = datetime.now(timezone.utc)
        limit_time = now + timedelta(minutes=NEWS_LOOKAHEAD_MIN)

        for evento in data:
            event_time_str = evento.get("Date") or evento.get("DateSpan")
            if not event_time_str:
                continue
            try:
                event_time = parser.isoparse(event_time_str).astimezone(timezone.utc)
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

def _guess_is_tree_model(m):
    """Heurística simple para saber si es árbol (LightGBM/XGBoost/sklearn trees)."""
    try:
        if hasattr(m, "feature_importances_"):
            return True
        if type(m).__name__.lower().startswith(("lgbm", "lightgbm", "xgb", "randomforest", "gradientboosting", "extratrees", "decisiontree")):
            return True
    except Exception:
        pass
    return False

def _build_shap_explainer(m, X_sample):
    """Crea el explainer SHAP si es posible y razonable."""
    if not HAS_SHAP:
        return None
    try:
        if _guess_is_tree_model(m):
            return shap.TreeExplainer(m)
        return None
    except Exception as e:
        logging.warning(f"No se pudo crear SHAP explainer: {e}")
        return None

def load_model():
    global model, feature_cols, last_model_time, explainer_shap, model_is_tree
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

            # ====== inicializar explainer SHAP si procede ======
            explainer_shap = None
            model_is_tree = _guess_is_tree_model(model)
            if model_is_tree and HAS_SHAP:
                logging.info("🧠 SHAP disponible — modelo tipo árbol detectado. Explainer se construirá on-demand.")
            else:
                if not HAS_SHAP:
                    logging.info("ℹ️ SHAP no instalado; seguirás viendo importancias globales si el modelo las expone.")
                else:
                    logging.info("ℹ️ Modelo no tipo árbol; se omite SHAP por rendimiento.")
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
    """Calcula indicadores y **descarta** filas sin datos (warm-up)."""
    if df.empty:
        return df
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    df["ema_9"]  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
    df["ema20"]  = ta.trend.EMAIndicator(close, window=20).ema_indicator()  # cache
    df["ema50"]  = ta.trend.EMAIndicator(close, window=50).ema_indicator()  # cache
    df["atr"]    = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    bb           = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    df["rsi"]    = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["adx"]    = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    return df.dropna()  # ✅ sin fillna(0.0)

# ====== utilidades de EXPLICABILIDAD ======
def _top_feature_importances(n=5):
    """Top N importancias globales del modelo (si existen)."""
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None or feature_cols is None:
            return []
        pairs = list(zip(feature_cols, map(float, importances)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = [(f, round(v, 6)) for f, v in pairs[:n] if v > 0]
        return top
    except Exception:
        return []

def _ensure_shap_explainer(X_for_fit=None):
    """Construye el explainer si aún no existe y es viable."""
    global explainer_shap
    if explainer_shap is not None:
        return explainer_shap
    if not HAS_SHAP or not model_is_tree:
        return None
    try:
        explainer_shap = shap.TreeExplainer(model)
        return explainer_shap
    except Exception as e:
        logging.warning(f"No se pudo crear SHAP explainer en _ensure_shap_explainer: {e}")
        explainer_shap = None
        return None

def build_local_explanation(X_row: pd.Series, class_idx: int, top_n=5):
    """
    Devuelve una explicación local por predicción:
    - top_importances: globales del modelo
    - shap_top: contribuciones locales (si SHAP disponible)
    """
    explanation = {
        "top_importances": _top_feature_importances(n=top_n),
        "shap_top": None
    }
    if X_row is None:
        return explanation

    # SHAP local
    try:
        expl = _ensure_shap_explainer()
        if expl is None:
            return explanation

        # Asegurar orden de columnas igual a feature_cols
        x = X_row.reindex(feature_cols).astype(float).values.reshape(1, -1)

        shap_values = expl.shap_values(x)
        # shap_values puede ser: array (binario) o lista por clase (multiclase)
        if isinstance(shap_values, list):
            # multiclase: elegir la clase target pedida
            if 0 <= class_idx < len(shap_values):
                sv = shap_values[class_idx].reshape(-1)  # (n_features,)
            else:
                sv = shap_values[0].reshape(-1)
        else:
            # binario/regresión
            sv = shap_values.reshape(-1)

        contribs = list(zip(feature_cols, np.abs(sv)))
        contribs.sort(key=lambda x: float(x[1]), reverse=True)
        top_local = [(f, round(float(val), 6)) for f, val in contribs[:top_n]]
        explanation["shap_top"] = top_local
        return explanation
    except Exception as e:
        logging.warning(f"No se pudo calcular SHAP local: {e}")
        return explanation

# ---------------- CEREBRO (condiciones globales) ----------------
def condiciones_de_mercado_ok_df(symbol, df_feat):
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
    if len(df_feat) < lookback:
        return None, None
    window = df_feat.iloc[-lookback:]
    high = float(window["High"].max())
    low = float(window["Low"].min())
    return low, high

def build_fib_levels(low, high):
    rng = high - low
    return {
        "0.0": low,
        "38.2": high - rng * 0.382,
        "50.0": high - rng * 0.500,
        "61.8": high - rng * 0.618,
        "100": high,
        "161.8ext_up": high + rng * 0.618,
        "161.8ext_dn": low  - rng * 0.618,
        "range": rng,
    }

def build_atr_plan(signal, price, atr):
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
    return {"plan": "ATR", "entry_suggest": round(entry, 4), "SL": round(sl, 4), "TP": round(tp, 4), "risk_rr": rr}

def compute_rr(entry, sl, tp):
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 1e-12:
            return None
        return round(reward / risk, 2)
    except Exception:
        return None

def choose_trade_plan(signal, df_feat):
    price = float(df_feat["Close"].iloc[-1])
    atr = float(df_feat["atr"].iloc[-1])
    adx = float(df_feat["adx"].iloc[-1])
    ema20 = float(df_feat["ema20"].iloc[-1])  # ✅ cache
    ema50 = float(df_feat["ema50"].iloc[-1])  # ✅ cache

    low, high = detect_swing_range(df_feat, FIB_LOOKBACK)
    if low is None or high is None or high <= low:
        return build_atr_plan(signal, price, atr)

    levels = build_fib_levels(low, high)
    rng_ratio = (levels["range"] / (price + 1e-9))
    use_fib = (rng_ratio >= MIN_RANGE_RATIO) and (adx >= MIN_ADX)

    if "COMPRA" in signal.upper() and use_fib and ema20 >= ema50:
        entry = float(levels["61.8"])
        sl = float(low + 0.0 - atr * SL_BUFFER_ATR_MULT)
        tp = float(levels["161.8ext_up"])
        plan = "FIB"
    elif "VENTA" in signal.upper() and use_fib and ema20 <= ema50:
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
        "risk_rr": rr
    }

# ---------------- NARRATIVA ----------------
def market_context_narrative(symbol, df, signal, prob_up_pct, prob_down_pct):
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
        narrative += f"📈 Prob ↑: {prob_up_pct:.2f}% | 📉 Prob ↓: {prob_down_pct:.2f}%\n"

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

# --------- util mapeo robusto de clases ----------
def get_class_index(classes, value):
    """Devuelve el índice de una clase concreta o arroja error claro si no existe."""
    if value not in classes:
        raise ValueError(f"Clase {value} no encontrada en modelo (clases={classes})")
    return classes.index(value)

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
        if df_feat.empty:
            return jsonify({"error": f"Datos insuficientes para {symbol} (warm-up)"}), 400

        X_all = df_feat.reindex(columns=feature_cols).astype(float)
        if X_all.empty:
            return jsonify({"error": "Features vacías"}), 500

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
                "narrative": f"🧠 Filtro estratégico activo: {why}",
                "explanation": None
            }
            if news:
                payload["news_event"] = news
            return jsonify(payload)

        preds_all = model.predict_proba(X_all.values)
        classes = list(getattr(model, "classes_", []))
        idx_up = get_class_index(classes, 1)
        idx_down = get_class_index(classes, -1)

        # ✅ Probabilidades internas en [0–1]
        prob_up = float(preds_all[-1, idx_up])
        prob_down = float(preds_all[-1, idx_down])

        # Señal IA
        signal = "ESPERAR"
        if (prob_up > UP_THRESHOLD) and (prob_up - prob_down >= DIFF_MARGIN):
            signal = "COMPRAR"
        elif (prob_down > DOWN_THRESHOLD) and (prob_down - prob_up >= DIFF_MARGIN):
            signal = "VENTA"

        # Confirmación técnica (con EMAs cacheadas)
        adx_val = float(df_feat["adx"].iloc[-1])
        ema20 = float(df_feat["ema20"].iloc[-1])
        ema50 = float(df_feat["ema50"].iloc[-1])
        confirmation = (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50) or \
                       (signal == "VENTA" and adx_val > 20 and ema20 < ema50)

        if signal == "COMPRAR":
            signal = "COMPRA CONFIRMADA ✅" if confirmation else "COMPRA POTENCIAL ⚠️"
        elif signal == "VENTA":
            signal = "VENTA CONFIRMADA ✅" if confirmation else "VENTA POTENCIAL ⚠️"

        price = float(df_feat["Close"].iloc[-1])
        trade_plan = choose_trade_plan(signal, df_feat)
        narrativa = market_context_narrative(symbol, df_feat, signal, prob_up*100, prob_down*100)

        # Explicabilidad local
        X_row = X_all.iloc[-1]
        dominant_class_idx = idx_up if prob_up >= prob_down else idx_down
        explanation = build_local_explanation(X_row, class_idx=dominant_class_idx, top_n=5)

        resp = {
            "symbol": symbol,
            "signal": signal,
            "prob_up": round(prob_up * 100, 2),    # ✅ solo para mostrar
            "prob_down": round(prob_down * 100, 2),
            "price": round(price, 2),
            "entry_suggest": trade_plan.get("entry_suggest"),
            "SL": trade_plan.get("SL"),
            "TP": trade_plan.get("TP"),
            "risk_rr": trade_plan.get("risk_rr"),
            "plan": trade_plan.get("plan"),
            "narrative": narrativa,
            "explanation": explanation
        }
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
            if df_feat.empty:
                continue

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

            X_all = df_feat.reindex(columns=feature_cols).astype(float)
            if X_all.empty:
                continue

            preds_all = model.predict_proba(X_all.values)
            classes = list(getattr(model, "classes_", []))
            idx_up = get_class_index(classes, 1)
            idx_down = get_class_index(classes, -1)

            prob_up = float(preds_all[-1, idx_up])   # [0–1]
            prob_down = float(preds_all[-1, idx_down])

            signal = "ESPERAR"
            if (prob_up > UP_THRESHOLD) and (prob_up - prob_down >= DIFF_MARGIN):
                signal = "COMPRAR"
            elif (prob_down > DOWN_THRESHOLD) and (prob_down - prob_up >= DIFF_MARGIN):
                signal = "VENTA"

            if signal in ("COMPRAR", "VENTA"):
                adx_val = float(df_feat["adx"].iloc[-1])
                ema20 = float(df_feat["ema20"].iloc[-1])
                ema50 = float(df_feat["ema50"].iloc[-1])
                confirm = (
                    (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50)
                    or
                    (signal == "VENTA" and adx_val > 20 and ema20 < ema50)
                )
                signal_out = signal + (" CONFIRMADA ✅" if confirm else " POTENCIAL ⚠️")
            else:
                signal_out = "ESPERAR"

            price = float(df_feat["Close"].iloc[-1])
            plan = choose_trade_plan(signal_out, df_feat)

            item = {
                "symbol": sym,
                "signal": signal_out,
                "prob_up": round(prob_up * 100, 2),     # mostrar en %
                "prob_down": round(prob_down * 100, 2),
                "price": round(price, 4),
                "entry_suggest": plan.get("entry_suggest"),
                "SL": plan.get("SL"),
                "TP": plan.get("TP"),
                "risk_rr": plan.get("risk_rr"),
                "plan": plan.get("plan")
                # (No agrego explanation aquí para no afectar performance del scanner)
            }
            signals_list.append(item)

        return jsonify(signals_list)

    except Exception as e:
        logging.exception("Error en /api/signals")
        return jsonify({"error": str(e)}), 500

# ---------------- SCANNER GLOBAL — PASO 6 ----------------
# Acepta /api/scanner y /api/scanner/
@app.get("/api/scanner")
@app.get("/api/scanner/")
def api_scanner():
    try:
        oportunidades = []
        for sym in SYMBOLS:
            df = download_klines_safe(sym)
            if df.empty:
                continue

            df_feat = compute_indicators(df)
            if df_feat.empty:
                continue

            ok, why, news = condiciones_de_mercado_ok_df(sym, df_feat)
            if not ok:
                continue

            X_all = df_feat.reindex(columns=feature_cols).astype(float)
            if X_all.empty:
                continue

            preds_all = model.predict_proba(X_all.values)
            classes = list(getattr(model, "classes_", []))
            idx_up = get_class_index(classes, 1)
            idx_down = get_class_index(classes, -1)
            prob_up = float(preds_all[-1, idx_up])      # [0–1]
            prob_down = float(preds_all[-1, idx_down])  # [0–1]

            # Señal IA
            signal = "ESPERAR"
            if (prob_up > UP_THRESHOLD) and (prob_up - prob_down >= DIFF_MARGIN):
                signal = "COMPRAR"
            elif (prob_down > DOWN_THRESHOLD) and (prob_down - prob_up >= DIFF_MARGIN):
                signal = "VENTA"

            # Confirmación técnica
            if signal in ("COMPRAR", "VENTA"):
                adx_val = float(df_feat["adx"].iloc[-1])
                ema20 = float(df_feat["ema20"].iloc[-1])
                ema50 = float(df_feat["ema50"].iloc[-1])
                confirm = (
                    (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50)
                    or
                    (signal == "VENTA" and adx_val > 20 and ema20 < ema50)
                )
                signal_out = signal + (" CONFIRMADA ✅" if confirm else " POTENCIAL ⚠️")
            else:
                continue

            price = float(df_feat["Close"].iloc[-1])
            plan = choose_trade_plan(signal_out, df_feat)
            rr = plan.get("risk_rr", 0)

            oportunidades.append({
                "symbol": sym,
                "signal": signal_out,
                "prob_up": round(prob_up * 100, 2),
                "prob_down": round(prob_down * 100, 2),
                "price": round(price, 4),
                "entry_suggest": plan.get("entry_suggest"),
                "SL": plan.get("SL"),
                "TP": plan.get("TP"),
                "risk_rr": rr,
                "plan": plan.get("plan"),
                "score": round((rr or 0) * 10, 2)
            })

        oportunidades.sort(key=lambda x: x["risk_rr"] or 0, reverse=True)
        return jsonify(oportunidades[:3])
    except Exception as e:
        logging.exception("Error en /api/scanner")
        return jsonify({"error": str(e)}), 500

# (Opcional) Ruta para ver qué rutas están registradas
@app.route("/_routes")
def _routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

# ===================================================
# 🧠 PASO 3 — Razonamiento multi-paso + simulación (What-If)
# ===================================================

def simulate_what_if(symbol: str, adjustments: dict):
    """
    Simula cómo cambiaría la predicción si ciertos indicadores se modificaran.
    adjustments: {"rsi": 70, "atr": 0.001, ...}
    """
    try:
        if not adjustments:
            logging.info(f"⚠️ No se aplicó razonamiento hipotético para {symbol}.")
            return {"symbol": symbol, "message": "Sin ajustes: no se aplicó razonamiento."}

        logging.info(f"🧠 Aplicando razonamiento hipotético para {symbol}: {adjustments}")
        df = download_klines_safe(symbol)
        if df.empty:
            return {"error": f"No hay datos para {symbol}"}

        df_feat = compute_indicators(df)
        if df_feat.empty:
            return {"error": f"Datos insuficientes para {symbol} (warm-up)"}

        X_all = df_feat.reindex(columns=feature_cols).astype(float)
        X_row = X_all.iloc[-1].copy()

        # Aplicar cambios simulados
        for feat, new_val in adjustments.items():
            if feat in X_row.index:
                X_row[feat] = float(new_val)

        # Pasar por el modelo
        probs = model.predict_proba(X_row.values.reshape(1, -1))
        classes = list(getattr(model, "classes_", []))
        idx_up = get_class_index(classes, 1)
        idx_down = get_class_index(classes, -1)

        prob_up = float(probs[0, idx_up])      # [0–1]
        prob_down = float(probs[0, idx_down])  # [0–1]

        original_probs = model.predict_proba(X_all.values)
        original_up = float(original_probs[-1, idx_up])
        original_down = float(original_probs[-1, idx_down])

        logging.info(f"📊 Resultado What-If {symbol} → UP: {prob_up*100:.2f}% | DOWN: {prob_down*100:.2f}%")

        return {
            "symbol": symbol,
            "message": "🧠 Se aplicó razonamiento hipotético",
            "adjustments": adjustments,
            "prob_up_simulated": round(prob_up * 100, 2),
            "prob_down_simulated": round(prob_down * 100, 2),
            "original_up": round(original_up * 100, 2),
            "original_down": round(original_down * 100, 2)
        }

    except Exception as e:
        logging.exception("Error en simulate_what_if")
        return {"error": str(e)}

# ---------------- NUEVA RUTA ----------------
@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """
    Ejemplo de request JSON:
    {
        "symbol": "BTCUSDT",
        "adjustments": {"rsi": 70, "atr": 0.001}
    }
    """
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").upper().strip()
        adjustments = data.get("adjustments") or {}

        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400
        if not adjustments:
            return jsonify({"error": "No se proporcionaron ajustes"}), 400

        result = simulate_what_if(symbol, adjustments)
        return jsonify(result)
    except Exception as e:
        logging.exception("Error en /api/whatif")
        return jsonify({"error": str(e)}), 500

# ---------------- MAIN ----------------
if __name__ == "__main__":
    client = init_client(API_KEY, API_SECRET)
    if client is None:
        logging.error("No se pudo conectar a Binance. Saliendo.")
        exit(1)
    app.run(host="0.0.0.0", port=5000, debug=False)
