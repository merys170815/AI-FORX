# ===============================
# 🤖 Trading IA Pro — PRO EDITION (API + UI mínima)
# ✅ Explicabilidad (SHAP + narrativa), umbrales por símbolo, Plan B, riesgo avanzado, backtest, what-if, scanner
# 🚫 No abre operaciones en Binance (solo analiza/sugiere)
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
from flask import Flask, jsonify, request, Response
from binance.client import Client
from datetime import datetime, timedelta, timezone
from dateutil import parser
from typing import Dict, Any, Tuple, List, Optional
from math import isnan
from flask import Flask, jsonify, request, Response, render_template, redirect, url_for

app = Flask(__name__)




# ====== 👇 SHAP opcional ======
try:
    import shap
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

# Umbrales base (se usan como defaults si no hay por-símbolo aún)
BASE_UP_THRESHOLD = 0.55
BASE_DOWN_THRESHOLD = 0.55
BASE_DIFF_MARGIN = 0.05

# Filtro estratégico
MIN_ATR_RATIO = 0.0003
MIN_ADX = 15

# FIB/ATR
FIB_LOOKBACK = 60
SL_BUFFER_ATR_MULT = 0.2
MIN_RANGE_RATIO = 0.003

# Noticias
TE_API_KEY = os.getenv("TE_API_KEY") or ""
NEWS_LOOKAHEAD_MIN = 30

# Riesgo
DAILY_R_MAX = 2.0  # máximo R a perder por día (suma de pérdidas simuladas)
DEFAULT_BALANCE = 10000.0
DEFAULT_RISK_PER_TRADE = 0.01  # 1% si no se usa Kelly
KELLY_FRACTION = 0.25  # Kelly acotado

# Ajuste dinámico
MAX_THRESHOLD = 0.80
MIN_THRESHOLD = 0.40
THRESHOLD_STEP = 0.01
HISTORY_CAP = 600  # por símbolo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

# Estado global
explainer_shap = None
model_is_tree = False
model = None
feature_cols: Optional[List[str]] = None
last_model_time = None

# ========= Helpers de thresholds JSON =========
def _ensure_thresholds_file():
    if not os.path.exists(THRESHOLD_STATE_FILE):
        with open(THRESHOLD_STATE_FILE, "w") as f:
            json.dump({"history": {}, "thresholds": {}}, f, indent=2)

def _read_state() -> Dict[str, Any]:
    _ensure_thresholds_file()
    with open(THRESHOLD_STATE_FILE, "r") as f:
        data = json.load(f)
    data.setdefault("history", {})
    data.setdefault("thresholds", {})
    return data

def _write_state(data: Dict[str, Any]):
    with open(THRESHOLD_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_symbol_thresholds(sym: str) -> Dict[str, float]:
    """
    Devuelve thresholds por símbolo; si no hay, usa base.
    """
    data = _read_state()
    th = data["thresholds"].get(sym, {})
    return {
        "UP": float(th.get("UP", BASE_UP_THRESHOLD)),
        "DOWN": float(th.get("DOWN", BASE_DOWN_THRESHOLD)),
        "DIFF": float(th.get("DIFF", BASE_DIFF_MARGIN)),
    }

def set_symbol_thresholds(sym: str, up: float, down: float, diff: float):
    data = _read_state()
    data["thresholds"][sym] = {
        "UP": max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(up))),
        "DOWN": max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(down))),
        "DIFF": max(0.0, min(0.25, float(diff))),  # acotamos diff
    }
    _write_state(data)

def guardar_resultado_trade(sym: str, signal: str, prob: float, result_win: bool, r_multiple: float):
    """
    Guarda resultado por símbolo para ajustar umbrales.
    """
    try:
        data = _read_state()
        hist = data["history"].get(sym, [])
        hist.append({
            "signal": signal,
            "prob": float(prob),
            "win": bool(result_win),
            "R": float(r_multiple),
            "timestamp": datetime.utcnow().isoformat()
        })
        hist = hist[-HISTORY_CAP:]
        data["history"][sym] = hist
        _write_state(data)
    except Exception as e:
        logging.warning(f"⚠️ Error guardando resultado: {e}")

def recalcular_thresholds_symbol(sym: str):
    """
    Recalcula dinámicamente los umbrales para un símbolo en función de su winrate local.
    """
    try:
        data = _read_state()
        hist = data["history"].get(sym, [])
        if len(hist) < 20:
            return  # aún no
        wins = sum(1 for h in hist if h.get("win"))
        total = len(hist)
        winrate = wins / total if total else 0.0

        th = get_symbol_thresholds(sym)
        up, down, diff = th["UP"], th["DOWN"], th["DIFF"]

        if winrate < 0.5 and up < MAX_THRESHOLD:
            up += THRESHOLD_STEP
            down += THRESHOLD_STEP
            diff = min(0.15, diff + 0.005)  # subimos un poco el margen
        elif winrate > 0.7 and up > MIN_THRESHOLD:
            up -= THRESHOLD_STEP
            down -= THRESHOLD_STEP
            diff = max(0.02, diff - 0.005)  # bajamos un poco el margen

        set_symbol_thresholds(sym, up, down, diff)
        logging.info(f"🔧 {sym}: thresholds → UP {up:.2f}, DOWN {down:.2f}, DIFF {diff:.3f}, winrate {winrate:.1%}")
    except Exception as e:
        logging.warning(f"⚠️ Error recalculando thresholds {sym}: {e}")

def auto_adjust_thresholds_loop():
    while True:
        time.sleep(3600)
        try:
            for s in SYMBOLS:
                recalcular_thresholds_symbol(s)
        except Exception as e:
            logging.warning(f"⚠️ Auto-adjust loop error: {e}")

threading.Thread(target=auto_adjust_thresholds_loop, daemon=True).start()

# ---------------- Binance ----------------
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

# ---------------- Noticias económicas ----------------
def hay_noticia_importante_proxima():
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

# ---------------- Modelo ----------------
def _guess_is_tree_model(m):
    try:
        if hasattr(m, "feature_importances_"):
            return True
        if type(m).__name__.lower().startswith(("lgbm","lightgbm","xgb","randomforest","gradientboosting","extratrees","decisiontree")):
            return True
    except Exception:
        pass
    return False

def _build_shap_explainer(m):
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
            model_is_tree = _guess_is_tree_model(model)
            explainer_shap = _build_shap_explainer(model)
    except Exception as e:
        logging.exception(f"Fallo cargando modelo '{MODEL_FILE}': {e}")

load_model()
threading.Thread(target=lambda: (time.sleep(60), load_model()), daemon=True).start()

if feature_cols is None:
    # Default robusto
    feature_cols = [
        "ema_10","ema_20","ema_50","sma_50","sma_200","ama_cross",
        "momentum","logret","atr14","bb_pct","rsi14","stoch_k","stoch_d",
        "macd","macd_signal","obv","vpt"
    ]

# ---------------- Datos ----------------
def download_klines_safe(sym):
    try:
        kl = client.futures_klines(symbol=sym, interval=INTERVAL, limit=HISTORICAL_LIMIT)
        if not kl:
            return pd.DataFrame()
        df = pd.DataFrame(kl, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
            "Quote_asset_volume", "Number_of_trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = df[c].astype(float)
        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
        df.set_index("Open_time", inplace=True)
        return df.ffill().bfill()
    except Exception as e:
        logging.error(f"Error descargando datos de {sym}: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # EMA/SMA
    df["ema_10"] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ama_cross"] = df["ema_10"] - df["ema_20"]

    # Momentum/retornos
    df["momentum"] = close.diff()
    df["logret"] = np.log(close).diff()

    # Volatilidad y bandas
    df["atr14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)

    # Osciladores
    df["rsi14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Volumen
    obv = ta.volume.OnBalanceVolumeIndicator(close, vol)
    df["obv"] = obv.on_balance_volume()
    df["vpt"] = (vol * (close.diff() / (close.shift(1) + 1e-9))).cumsum()

    # Filtros (para cerebro)
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    # Lags para robustez (si el modelo los usa)
    for f in ["atr14","bb_pct","rsi14","stoch_k","stoch_d","macd","macd_signal","vpt","ama_cross","momentum","logret"]:
        for lag in (1, 2, 3):
            df[f"{f}_lag{lag}"] = df[f].shift(lag)

    return df.dropna()

# ---------------- Explicabilidad ----------------
def _top_feature_importances(n=5) -> List[Tuple[str, float]]:
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None or feature_cols is None:
            return []
        pairs = list(zip(feature_cols, map(float, importances)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(f, round(v, 6)) for f, v in pairs[:n] if v > 0]
    except Exception:
        return []

def _ensure_shap_explainer():
    global explainer_shap
    if explainer_shap is not None:
        return explainer_shap
    if not HAS_SHAP or not model_is_tree:
        return None
    try:
        explainer_shap = shap.TreeExplainer(model)
        return explainer_shap
    except Exception as e:
        logging.warning(f"No se pudo crear TreeExplainer: {e}")
        explainer_shap = None
        return None

def build_local_explanation(X_row: pd.Series, class_idx: int, top_n=5) -> Dict[str, Any]:
    explanation = {"top_importances": _top_feature_importances(n=top_n), "shap_top": None, "reason_text": None}
    if X_row is None:
        return explanation
    # SHAP local
    try:
        expl = _ensure_shap_explainer()
        if expl is not None:
            x = X_row.reindex(feature_cols).astype(float).values.reshape(1, -1)
            shap_values = expl.shap_values(x)
            if isinstance(shap_values, list):
                sv = shap_values[class_idx].reshape(-1)
            else:
                sv = shap_values.reshape(-1)
            contribs = list(zip(feature_cols, np.abs(sv)))
            contribs.sort(key=lambda x: float(x[1]), reverse=True)
            explanation["shap_top"] = [(f, round(float(val), 6)) for f, val in contribs[:top_n]]

        # Texto natural (heurística)
        explanation["reason_text"] = explain_like_human(X_row, explanation)
        return explanation
    except Exception as e:
        logging.warning(f"No se pudo calcular SHAP local: {e}")
        explanation["reason_text"] = explain_like_human(X_row, explanation)
        return explanation

def explain_like_human(xrow: pd.Series, exp: Dict[str, Any]) -> str:
    """
    Genera una frase legible tipo: "Compra impulsada por RSI alto y cruce EMA".
    Heurística si no hay SHAP.
    """
    hints = []
    try:
        rsi = xrow.get("rsi14", np.nan)
        ema10 = xrow.get("ema_10", np.nan)
        ema20 = xrow.get("ema_20", np.nan)
        ema50 = xrow.get("ema_50", np.nan)
        bbp = xrow.get("bb_pct", np.nan)
        adx = xrow.get("adx", np.nan) if "adx" in xrow else np.nan
        macd_v = xrow.get("macd", np.nan)
        macd_s = xrow.get("macd_signal", np.nan)

        if not isnan(rsi):
            if rsi >= 60: hints.append("RSI fuerte")
            elif rsi <= 40: hints.append("RSI débil")

        if not any(map(isnan, [ema10, ema20])):
            if ema10 > ema20: hints.append("EMAs en cruce alcista")
            elif ema10 < ema20: hints.append("EMAs en cruce bajista")

        if not any(map(isnan, [ema20, ema50])):
            if ema20 > ema50: hints.append("EMA20 por encima de EMA50 (tendencia)")
            elif ema20 < ema50: hints.append("EMA20 por debajo de EMA50 (tendencia débil)")

        if not isnan(bbp):
            if bbp >= 0.8: hints.append("Cerca de banda superior (posible sobrecompra)")
            elif bbp <= 0.2: hints.append("Cerca de banda inferior (posible sobreventa)")

        if not any(map(isnan, [macd_v, macd_s])):
            if macd_v > macd_s: hints.append("MACD positivo")
            elif macd_v < macd_s: hints.append("MACD negativo")

        if not isnan(adx):
            if adx >= 20: hints.append("ADX suficiente (tendencia)")
            else: hints.append("ADX bajo (poca tendencia)")

        if exp.get("shap_top"):
            top_feats = [f for f, _ in exp["shap_top"][:3]]
            hints.append("SHAP: " + ", ".join(top_feats))

        if not hints:
            return "Señal generada por combinación de momentum, tendencia y volatilidad."
        # Compactar en una frase:
        return " / ".join(hints[:4])
    except Exception:
        return "Señal generada por combinación de momentum, tendencia y volatilidad."

# ---------------- Filtros/Plan ----------------
def condiciones_de_mercado_ok_df(symbol, df_feat) -> Tuple[bool, str, Optional[Dict[str, str]]]:
    news = hay_noticia_importante_proxima()
    if news:
        return False, "Noticia de alto impacto próxima", news

    for col in ("Close", "atr", "adx"):
        if col not in df_feat.columns or df_feat[col].isna().iloc[-1]:
            logging.info(f"❌ Falta indicador: {col} en {symbol}")
            return False, "Datos insuficientes", None

    price = float(df_feat["Close"].iloc[-1])
    atr = float(df_feat["atr"].iloc[-1])
    adx = float(df_feat["adx"].iloc[-1])

    atr_ratio = atr / (price + 1e-9)
    if atr_ratio < MIN_ATR_RATIO:
        return False, f"Volatilidad baja (ATR/Precio={atr_ratio:.5f} < {MIN_ATR_RATIO})", None

    if adx < MIN_ADX:
        # Permitimos operar como "Potencial", pero el plan reflejará debilidad
        return True, f"Tendencia débil (ADX={adx:.2f} < {MIN_ADX})", None

    return True, "OK", None

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

def compute_rr(entry, sl, tp):
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 1e-12:
            return None
        return round(reward / risk, 2)
    except Exception:
        return None

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

def choose_trade_plan(signal, df_feat):
    price = float(df_feat["Close"].iloc[-1])
    atr = float(df_feat["atr"].iloc[-1])
    adx = float(df_feat["adx"].iloc[-1])
    ema20 = float(df_feat.get("ema_20", df_feat.get("ema20", np.nan)).iloc[-1])
    ema50 = float(df_feat.get("ema_50", df_feat.get("ema50", np.nan)).iloc[-1])

    low, high = detect_swing_range(df_feat, FIB_LOOKBACK)
    if low is None or high is None or high <= low:
        plan = build_atr_plan(signal, price, atr)
    else:
        levels = build_fib_levels(low, high)
        rng_ratio = (levels["range"] / (price + 1e-9))
        use_fib = (rng_ratio >= MIN_RANGE_RATIO) and (adx >= MIN_ADX)

        if "COMPRA" in signal.upper() and use_fib and not isnan(ema20) and not isnan(ema50) and ema20 >= ema50:
            entry = float(levels["61.8"])
            sl = float(low - atr * SL_BUFFER_ATR_MULT)
            tp = float(levels["161.8ext_up"])
            rr = compute_rr(entry, sl, tp)
            plan = {"plan": "FIB", "entry_suggest": round(entry, 4), "SL": round(sl, 4), "TP": round(tp, 4), "risk_rr": rr}
        elif "VENTA" in signal.upper() and use_fib and not isnan(ema20) and not isnan(ema50) and ema20 <= ema50:
            entry = float(levels["38.2"])
            sl = float(high + atr * SL_BUFFER_ATR_MULT)
            tp = float(levels["161.8ext_dn"])
            rr = compute_rr(entry, sl, tp)
            plan = {"plan": "FIB", "entry_suggest": round(entry, 4), "SL": round(sl, 4), "TP": round(tp, 4), "risk_rr": rr}
        else:
            plan = build_atr_plan(signal, price, atr)

    # Reglas Plan B (post-entrada): solo descripción/sugerencia aquí
    plan["planB_rules"] = {
        "if_adx_drop": "Si ADX cae < 15 tras 3 velas, cambiar a ATR (TP=1.2*ATR, SL=1.0*ATR desde entrada).",
        "if_big_counter": "Si vela en contra > 1.2*ATR, reducir 50% posición y activar trailing ATR 1x."
    }
    return plan

# ---------------- Señal/Narrativa ----------------
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

        context = f"📊 {symbol} — Precio: {price:.4f} | Prob↑ {prob_up_pct:.2f}% Prob↓ {prob_down_pct:.2f}% | Régimen: {regime}. "
        if atr_distance_high <= 1.0:
            context += f"Cerca resistencia {high_zone:.4f}. "
        elif atr_distance_low <= 1.0:
            context += f"Cerca soporte {low_zone:.4f}. "
        else:
            context += "Sin S/R inmediatos. "
        if "CONFIRMADA" in signal.upper():
            context += "Señal confirmada. "
        elif "POTENCIAL" in signal.upper():
            context += "Señal potencial (confirmación pendiente). "
        return context
    except Exception as e:
        logging.warning(f"Error narrativa {symbol}: {e}")
        return f"{symbol}: contexto no disponible."

# ---------------- Gestión de riesgo (sizing sugerido) ----------------
def kelly_sizing(winrate: float, rr: float, fraction_cap: float = KELLY_FRACTION) -> float:
    """
    Kelly teórica: f* = p - (1-p)/b  (b = RR)
    Capada a fraction_cap para prudencia. Devuelve fracción del capital.
    """
    try:
        if rr is None or rr <= 0:
            return 0.0
        p = max(0.0, min(1.0, float(winrate)))
        b = float(rr)
        f = p - (1 - p) / b
        return max(0.0, min(fraction_cap, f))
    except Exception:
        return 0.0

def atr_target_position(balance: float, risk_per_trade: float, entry: float, sl: float) -> float:
    """
    Tamaño (en unidades del activo) para arriesgar 'risk_per_trade' del balance.
    """
    try:
        risk_money = balance * risk_per_trade
        stop_distance = abs(entry - sl)
        if stop_distance <= 1e-12:
            return 0.0
        units = risk_money / stop_distance
        return max(0.0, units)
    except Exception:
        return 0.0

# ---------------- Predicción clase index ----------------
def get_class_index(classes, value):
    if value in classes:
        return classes.index(value)
    alt_map = {1: "UP", -1: "DOWN", "UP": 1, "DOWN": -1}
    alt_val = alt_map.get(value)
    if alt_val in classes:
        return classes.index(alt_val)
    raise ValueError(f"Clase {value} no encontrada en modelo (clases={classes})")

# ---------------- Core de señal ----------------
def compute_signal_for_symbol(symbol: str, balance: float = DEFAULT_BALANCE, use_kelly: bool = False) -> Dict[str, Any]:
    df = download_klines_safe(symbol)
    if df.empty:
        return {"error": f"No hay datos para {symbol}"}

    df_feat = compute_indicators(df)
    if df_feat.empty:
        return {"error": f"Datos insuficientes para {symbol} (warm-up)"}

    X_all = df_feat.reindex(columns=feature_cols).astype(float)
    if X_all.empty:
        return {"error": "Features vacías"}

    ok, why, news = condiciones_de_mercado_ok_df(symbol, df_feat)
    price_now = float(df_feat["Close"].iloc[-1])

    th = get_symbol_thresholds(symbol)
    preds_all = model.predict_proba(X_all.values)
    classes = list(getattr(model, "classes_", []))
    idx_up = get_class_index(classes, 1)
    idx_down = get_class_index(classes, -1)
    prob_up = float(preds_all[-1, idx_up])  # [0-1]
    prob_down = float(preds_all[-1, idx_down])

    # Señal IA base
    signal = "ESPERAR"
    if (prob_up > th["UP"]) and (prob_up - prob_down >= th["DIFF"]):
        signal = "COMPRAR"
    elif (prob_down > th["DOWN"]) and (prob_down - prob_up >= th["DIFF"]):
        signal = "VENTA"

    # Confirmación técnica
    adx_val = float(df_feat["adx"].iloc[-1])
    ema20 = float(df_feat.get("ema_20", df_feat.get("ema20", np.nan)).iloc[-1])
    ema50 = float(df_feat.get("ema_50", df_feat.get("ema50", np.nan)).iloc[-1])
    confirm = (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50) or \
              (signal == "VENTA" and adx_val > 20 and ema20 < ema50)

    if signal == "COMPRAR":
        signal_out = "COMPRA CONFIRMADA ✅" if confirm else "COMPRA POTENCIAL ⚠️"
    elif signal == "VENTA":
        signal_out = "VENTA CONFIRMADA ✅" if confirm else "VENTA POTENCIAL ⚠️"
    else:
        # Si el filtro estratégico no aprueba, marcamos esperar y explicamos
        msg = "Noticia de alto impacto próxima — Evitar operar." if news else f"Condiciones no ideales: {why}"
        return {
            "symbol": symbol, "signal": "ESPERAR 🧠" if not news else "ESPERAR 🚨",
            "message": msg, "price": round(price_now, 4),
            "prob_up": None, "prob_down": None,
            "entry_suggest": None, "SL": None, "TP": None, "risk_rr": None,
            "plan": "NONE", "narrative": f"🧠 Filtro estratégico: {why}", "explanation": None,
            **({"news_event": news} if news else {})
        }

    # Plan de trade (incluye planB reglas)
    trade_plan = choose_trade_plan(signal_out, df_feat)

    # Explicabilidad local
    X_row = X_all.iloc[-1]
    dominant_class_idx = idx_up if prob_up >= prob_down else idx_down
    explanation = build_local_explanation(X_row, class_idx=dominant_class_idx, top_n=5)

    # Narrativa
    narrativa = market_context_narrative(symbol, df_feat, signal_out, prob_up*100, prob_down*100)

    # Sizing sugerido (no ejecuta trades)
    entry = trade_plan.get("entry_suggest")
    sl = trade_plan.get("SL")
    rr = trade_plan.get("risk_rr")

    sizing = None
    if entry and sl:
        if use_kelly:
            # Kelly usando winrate aproximado a partir del historial del símbolo
            data = _read_state()
            hist = data["history"].get(symbol, [])
            if len(hist) >= 25:
                wins = sum(1 for h in hist if h.get("win"))
                wr = wins / len(hist)
            else:
                wr = 0.52  # suposición neutra
            frac = kelly_sizing(wr, rr or 1.2, fraction_cap=KELLY_FRACTION)
            units = atr_target_position(balance, frac, entry, sl)
            sizing = {"method": "Kelly_fraccional", "kelly_wr": round(wr, 3), "fraction": round(frac, 4), "units": round(units, 6)}
        else:
            units = atr_target_position(balance, DEFAULT_RISK_PER_TRADE, entry, sl)
            sizing = {"method": "ATR_targeting", "risk_per_trade": DEFAULT_RISK_PER_TRADE, "units": round(units, 6)}

    resp = {
        "symbol": symbol,
        "signal": signal_out,
        "prob_up": round(prob_up * 100, 2),
        "prob_down": round(prob_down * 100, 2),
        "price": round(price_now, 4),
        "entry_suggest": trade_plan.get("entry_suggest"),
        "SL": trade_plan.get("SL"),
        "TP": trade_plan.get("TP"),
        "risk_rr": trade_plan.get("risk_rr"),
        "plan": trade_plan.get("plan"),
        "planB_rules": trade_plan.get("planB_rules"),
        "narrative": narrativa,
        "explanation": explanation,
        "sizing": sizing,
        "thresholds_used": th
    }
    return resp

# ---------------- Backtest ----------------
def simulate_trade(row_entry_idx, side, entry, sl, tp, df_feat, use_planB: bool) -> Tuple[float, int]:
    """
    Simula trade en adelante desde row_entry_idx+1.
    Retorna (R_multiple, bars_held).
    Si use_planB=True, aplica reglas: ADX<15 o vela en contra >1.2*ATR -> reduce/exita/trailing.
    """
    bars = 0
    risk = abs(entry - sl)
    if risk <= 1e-12:
        return 0.0, 0

    size_factor = 1.0  # para salida parcial en planB
    trailing_sl = None

    for i in range(row_entry_idx + 1, len(df_feat)):
        bars += 1
        c = float(df_feat["Close"].iloc[i])
        h = float(df_feat["High"].iloc[i])
        l = float(df_feat["Low"].iloc[i])
        atr = float(df_feat["atr"].iloc[i])
        adx = float(df_feat["adx"].iloc[i])

        # Plan B
        if use_planB:
            # ADX drop
            if adx < 15 and trailing_sl is None:
                # Convertimos a ATR-plan con TP suavecillo
                if side == "long":
                    trailing_sl = c - 1.0 * atr
                    tp_alt = c + 1.2 * atr
                    # si llega al tp_alt, cerramos
                    if h >= tp_alt:
                        gain = (tp_alt - entry) / risk
                        return round(gain * size_factor, 2), bars
                else:
                    trailing_sl = c + 1.0 * atr
                    tp_alt = c - 1.2 * atr
                    if l <= tp_alt:
                        gain = (entry - tp_alt) / risk
                        return round(gain * size_factor, 2), bars
            # vela fuerte en contra > 1.2 ATR
            rng_bar = h - l
            if ((side == "long" and (entry - l) > 1.2 * atr) or
                (side == "short" and (h - entry) > 1.2 * atr)) and size_factor > 0.51:
                size_factor = 0.5  # salida parcial 50%

        # trailing si existe
        if trailing_sl is not None:
            if side == "long":
                trailing_sl = max(trailing_sl, c - 1.0 * atr)
                if l <= trailing_sl:
                    gain = (trailing_sl - entry) / risk
                    return round(gain * size_factor, 2), bars
            else:
                trailing_sl = min(trailing_sl, c + 1.0 * atr)
                if h >= trailing_sl:
                    gain = (entry - trailing_sl) / risk
                    return round(gain * size_factor, 2), bars

        # Check TP/SL originales
        if side == "long":
            if h >= tp:
                gain = (tp - entry) / risk
                return round(gain, 2), bars
            if l <= sl:
                loss = (sl - entry) / risk
                return round(loss, 2), bars
        else:
            if l <= tp:
                gain = (entry - tp) / risk
                return round(gain, 2), bars
            if h >= sl:
                loss = (entry - sl) / risk
                return round(loss, 2), bars

    # si no tocó nada, cerramos al final a mercado
    last_close = float(df_feat["Close"].iloc[-1])
    if side == "long":
        res = (last_close - entry) / risk
    else:
        res = (entry - last_close) / risk
    return round(res, 2), bars

def backtest_symbol(symbol: str, days: int = 60, use_planB: bool = True) -> Dict[str, Any]:
    df = download_klines_safe(symbol)
    if df.empty:
        return {"error": f"No hay datos para {symbol}"}
    # recortar últimos N días
    since = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= since]
    df_feat = compute_indicators(df)
    if df_feat.empty or len(df_feat) < 200:
        return {"error": "Datos insuficientes para backtest"}

    X_all = df_feat.reindex(columns=feature_cols).astype(float)
    preds = model.predict_proba(X_all.values)
    classes = list(getattr(model, "classes_", []))
    idx_up = get_class_index(classes, 1)
    idx_down = get_class_index(classes, -1)

    th = get_symbol_thresholds(symbol)
    results = []
    cum_R = 0.0
    day_loss_R: Dict[str, float] = {}  # control de pérdidas por día

    for i in range(1, len(df_feat) - 1):
        prob_up = float(preds[i, idx_up])
        prob_down = float(preds[i, idx_down])

        # señal
        signal = "ESPERAR"
        if (prob_up > th["UP"]) and (prob_up - prob_down >= th["DIFF"]):
            signal = "COMPRAR"
        elif (prob_down > th["DOWN"]) and (prob_down - prob_up >= th["DIFF"]):
            signal = "VENTA"
        if signal == "ESPERAR":
            continue

        # Confirmación
        adx_val = float(df_feat["adx"].iloc[i])
        ema20 = float(df_feat.get("ema_20", df_feat.get("ema20", np.nan)).iloc[i])
        ema50 = float(df_feat.get("ema_50", df_feat.get("ema50", np.nan)).iloc[i])
        confirm = (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50) or \
                  (signal == "VENTA" and adx_val > 20 and ema20 < ema50)
        signal_out = signal + (" CONFIRMADA ✅" if confirm else " POTENCIAL ⚠️")

        # plan
        sub_df = df_feat.iloc[: i + 1]  # hasta i
        plan = choose_trade_plan(signal_out, sub_df)
        entry, sl, tp = plan.get("entry_suggest"), plan.get("SL"), plan.get("TP")
        if not entry or not sl or not tp or plan.get("plan") == "NONE":
            continue

        # tope de pérdida diaria por R
        day_key = df_feat.index[i].strftime("%Y-%m-%d")
        lost_today = day_loss_R.get(day_key, 0.0)
        if lost_today <= -DAILY_R_MAX:
            continue  # saltamos señales si ya alcanzamos tope de pérdida

        side = "long" if "COMPRA" in signal_out.upper() else "short"
        R, bars = simulate_trade(i, side, entry, sl, tp, df_feat, use_planB=use_planB)
        cum_R += R
        if R < 0:
            day_loss_R[day_key] = day_loss_R.get(day_key, 0.0) + R

        results.append({"idx": i, "time": str(df_feat.index[i]), "signal": signal_out,
                        "entry": entry, "SL": sl, "TP": tp, "R": R, "bars": bars})

    if not results:
        return {"symbol": symbol, "days": days, "trades": 0, "cum_R": 0.0, "winrate": None, "avgR": None, "maxDD_R": 0.0, "details": []}

    Rs = [r["R"] for r in results]
    wins = sum(1 for r in Rs if r > 0)
    cum_curve = np.cumsum(Rs)
    max_dd = 0.0
    peak = 0.0
    for v in cum_curve:
        if v > peak:
            peak = v
        dd = peak - v
        max_dd = max(max_dd, dd)

    return {
        "symbol": symbol,
        "days": days,
        "trades": len(results),
        "cum_R": round(float(np.sum(Rs)), 2),
        "winrate": round(wins / len(results), 3),
        "avgR": round(float(np.mean(Rs)), 3),
        "maxDD_R": round(float(max_dd), 2),
        "details": results[:200]  # limita respuesta
    }

# ---------------- Rutas ----------------

# ===================================================
# 🌐 RUTAS PRINCIPALES
# ===================================================

# 👉 Dashboard principal (Frontend UI)
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# 👉 Página raíz: redirige automáticamente al dashboard
@app.route("/")
def home():
    return redirect(url_for("dashboard"))

# 👉 Página informativa de endpoints (antes era la ruta raíz)
@app.route("/api-info")
def api_info():
    html = """
<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Trading IA Pro — API</title>
<style>
body{font-family:Arial;background:#0d1117;color:#e6edf3;padding:30px}
a{color:#58a6ff}
</style>
</head><body>
<h1>🚀 Trading IA Pro — API</h1>
<p>Endpoints útiles:</p>
<ul>
<li>POST <code>/api/ask</code> — body: {"symbol":"BTCUSDT","balance":10000,"use_kelly":false}</li>
<li>GET  <code>/api/scanner</code></li>
<li>POST <code>/api/whatif</code> — body: {"symbol":"BTCUSDT","adjustments":{"rsi14":65}}</li>
<li>POST <code>/api/backtest</code> — body: {"symbol":"BTCUSDT","days":60,"planB":true}</li>
<li>GET  <code>/_thresholds?symbol=BTCUSDT</code></li>
</ul>
<p>⚠️ Este servidor <b>no</b> ejecuta órdenes. Solo análisis y simulación.</p>
</body></html>
    """
    return Response(html, mimetype="text/html")


@app.route("/_thresholds")
def get_thresholds_view():
    sym = (request.args.get("symbol") or "").upper().strip()
    if not sym:
        return jsonify({"error": "symbol requerido"}), 400
    return jsonify({"symbol": sym, "thresholds": get_symbol_thresholds(sym)})

@app.route("/api/ask", methods=["POST"])
def api_ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        symbol = (data.get("symbol") or "").upper().strip()
        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400
        balance = float(data.get("balance", DEFAULT_BALANCE))
        use_kelly = bool(data.get("use_kelly", False))
        resp = compute_signal_for_symbol(symbol, balance=balance, use_kelly=use_kelly)
        return jsonify(resp)
    except Exception as e:
        logging.exception("Error en /api/ask")
        return jsonify({"error": str(e)}), 500

@app.get("/api/scanner")
def api_scanner():
    try:
        oportunidades = []
        for sym in SYMBOLS:
            result = compute_signal_for_symbol(sym)
            if result.get("error"):
                continue
            # Solo oportunidades activas
            sig = result.get("signal", "")
            if "COMPRA" in sig or "VENTA" in sig:
                rr = result.get("risk_rr") or 0
                oportunidades.append({
                    "symbol": sym,
                    "signal": sig,
                    "prob_up": result.get("prob_up"),
                    "prob_down": result.get("prob_down"),
                    "price": result.get("price"),
                    "entry_suggest": result.get("entry_suggest"),
                    "SL": result.get("SL"),
                    "TP": result.get("TP"),
                    "risk_rr": rr,
                    "plan": result.get("plan")
                })
        if not oportunidades:
            return jsonify([])
        oportunidades.sort(key=lambda x: x["risk_rr"] or 0, reverse=True)
        return jsonify(oportunidades[:3])
    except Exception as e:
        logging.exception("Error en /api/scanner")
        return jsonify({"error": str(e)}), 500

@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    try:
        data = request.get_json(force=True, silent=True) or {}
        symbol = (data.get("symbol") or "").upper().strip()
        adjustments = data.get("adjustments") or {}
        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400
        if not adjustments:
            return jsonify({"error": "No se proporcionaron ajustes"}), 400

        df = download_klines_safe(symbol)
        if df.empty:
            return jsonify({"error": f"No hay datos para {symbol}"})
        df_feat = compute_indicators(df)
        if df_feat.empty:
            return jsonify({"error": f"Datos insuficientes para {symbol} (warm-up)"})
        X_all = df_feat.reindex(columns=feature_cols).astype(float)
        X_row = X_all.iloc[-1].copy()

        invalid_feats = [f for f in adjustments if f not in X_row.index]
        if invalid_feats:
            return jsonify({"error": f"Features inválidos: {invalid_feats}. Usa nombres exactos de feature."}), 400

        for feat, new_val in adjustments.items():
            X_row[feat] = float(new_val)

        probs = model.predict_proba(X_row.values.reshape(1, -1))
        classes = list(getattr(model, "classes_", []))
        idx_up = get_class_index(classes, 1)
        idx_down = get_class_index(classes, -1)

        prob_up = float(probs[0, idx_up])
        prob_down = float(probs[0, idx_down])

        original_probs = model.predict_proba(X_all.values)
        original_up = float(original_probs[-1, idx_up])
        original_down = float(original_probs[-1, idx_down])

        return jsonify({
            "symbol": symbol,
            "message": "🧠 What-If aplicado",
            "adjustments": adjustments,
            "prob_up_simulated": round(prob_up * 100, 2),
            "prob_down_simulated": round(prob_down * 100, 2),
            "original_up": round(original_up * 100, 2),
            "original_down": round(original_down * 100, 2)
        })
    except Exception as e:
        logging.exception("Error en /api/whatif")
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    try:
        data = request.get_json(force=True, silent=True) or {}
        symbol = (data.get("symbol") or "").upper().strip()
        days = int(data.get("days", 60))
        use_planB = bool(data.get("planB", True))
        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400
        resB = backtest_symbol(symbol, days=days, use_planB=use_planB)
        resA = backtest_symbol(symbol, days=days, use_planB=False)  # sin plan B para comparar
        return jsonify({"with_planB": resB, "without_planB": resA})
    except Exception as e:
        logging.exception("Error en /api/backtest")
        return jsonify({"error": str(e)}), 500

# ===================================================
# 🧰 MANEJO GLOBAL DE ERRORES
# ===================================================
from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("❌ Error no controlado")
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description, "status": e.code}), e.code
    return jsonify({"error": str(e), "status": 500}), 500
# ===================================================
# 📊 NUEVO: Estadísticas globales del sistema
# ===================================================
@app.route("/api/stats")
def api_stats():
    try:
        file_path = "backtest_results_optim.csv"

        # 📌 Verificar que el archivo exista
        if not os.path.exists(file_path):
            return jsonify({"error": "No se encontró el archivo backtest_results_optim.csv"}), 404

        df = pd.read_csv(file_path)

        # 📊 Total de trades (o filas)
        total_trades = len(df)

        # ✅ Manejo flexible de columnas
        if 'result' in df.columns:
            wins = df[df['result'].astype(str).str.lower() == 'win'].shape[0]
            losses = df[df['result'].astype(str).str.lower() == 'loss'].shape[0]
        else:
            # Si no hay columna 'result', asumimos que cada fila es un trade
            # y que las ganancias se pueden aproximar desde accuracy si existe
            if 'accuracy' in df.columns and total_trades > 0:
                # si accuracy está en %, lo convertimos a proporción
                avg_acc = df['accuracy'].mean()
                wins = int((avg_acc / 100) * total_trades) if avg_acc > 1 else int(avg_acc * total_trades)
            else:
                wins = total_trades  # por defecto todos ganadores
            losses = total_trades - wins

        # 🧮 Winrate
        winrate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0

        # ⚖️ R:R promedio
        if 'profit_factor' in df.columns:
            avg_rr = round(df['profit_factor'].replace([float('inf'), -float('inf')], 0).mean(), 2)
        elif 'rr' in df.columns:
            avg_rr = round(df['rr'].mean(), 2)
        else:
            avg_rr = 0

        # 📉 Máximo drawdown
        if 'drawdown' in df.columns:
            max_dd = round(df['drawdown'].max(), 2)
        else:
            max_dd = 0

        # 📤 Respuesta final en JSON
        return jsonify({
            "total_trades": int(total_trades),
            "wins": int(wins),
            "losses": int(losses),
            "winrate": winrate,
            "avg_rr": avg_rr,
            "max_drawdown": max_dd
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ===================================================
# 🚀 MAIN
# ===================================================
if __name__ == "__main__":
    _ensure_thresholds_file()
    client = init_client(API_KEY, API_SECRET)
    if client is None:
        logging.error("No se pudo conectar a Binance. Saliendo.")
        exit(1)
    app.run(host="0.0.0.0", port=5000, debug=False)

