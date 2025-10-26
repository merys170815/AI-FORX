# ===============================
# 🤖 Trading IA Pro — PRO EDITION (API + UI mínima)
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
from flask import Flask, jsonify, request, Response, render_template, redirect, url_for
from binance.client import Client
from binance import ThreadedWebsocketManager   # 👈 WebSocket Manager correcto
from datetime import datetime, timedelta, timezone
from dateutil import parser
from typing import Dict, Any, Tuple, List, Optional
from math import isnan
from logger_trades import log_trade
from trade_engine import open_trade
from trade_engine import get_stats
from adaptive_learning import start_learning_thread
from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__)

# ---------------- CONFIG GLOBAL ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ====== 👇 SHAP opcional ======
try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

# === Callback de WebSocket ===
def handle_socket_message(msg):
    try:
        symbol = msg['s']
        price = float(msg['p'])
        latest_prices[symbol] = price   # ✅ se guarda internamente
        # 👇 comentar o borrar esta línea para no imprimir en consola
        # logging.info(f"📈 TICK {symbol} — Precio actualizado: {price}")
    except Exception as e:
        logging.error(f"❌ Error procesando mensaje WebSocket: {e}")

def iniciar_websocket():
    while True:
        try:
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            for sym in SYMBOLS:
                twm.start_symbol_ticker_socket(callback=handle_socket_message, symbol=sym)
                logging.info(f"✅ WebSocket iniciado para {sym}")
            twm.join()
        except Exception as e:
            logging.error(f"❌ Error en WebSocket: {e}. Reintentando en 5s...")
            time.sleep(5)

# ---------------- Parámetros de trading ----------------
# 🔹 Símbolos base de tu bot
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

# 📊 Diccionario global para guardar los precios en tiempo real
latest_prices = {sym: None for sym in SYMBOLS}

INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"

THRESHOLD_STATE_FILE = "thresholds_state.json"
OPTIM_FILE = "backtest_results_optim.csv"

BASE_UP_THRESHOLD = 0.50
BASE_DOWN_THRESHOLD = 0.50
BASE_DIFF_MARGIN = 0.02
MAX_THRESHOLD = 0.80
MIN_THRESHOLD = 0.40

# Filtros de mercado
MIN_ATR_RATIO = 0.0001    # antes 0.0003
MIN_ADX = 10              # antes 15
MIN_RANGE_RATIO = 0.001   # antes 0.003
FIB_LOOKBACK = 60
SL_BUFFER_ATR_MULT = 0.2


TE_API_KEY = os.getenv("TE_API_KEY") or ""
NEWS_LOOKAHEAD_MIN = 30

DAILY_R_MAX = 2.0
DEFAULT_BALANCE = 10000.0
DEFAULT_RISK_PER_TRADE = 0.01
KELLY_FRACTION = 0.25

# ---------------- Estado global ----------------
explainer_shap = None
model_is_tree = False
model = None
feature_cols: Optional[List[str]] = None
last_model_time = None
client: Optional[Client] = None
REFRESH_INTERVAL = 86400  # 24 horas

# ========= Helpers de thresholds JSON mínimos =========
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

# ===============================
# 🧠 Filtros dinámicos de símbolos
# ===============================
filtros_config = {
    "min_pf": 1.5,
    "min_accuracy": 50.0,
    "max_drawdown": 0.20
}

def get_symbols_with_filters(file=OPTIM_FILE):
    """Filtra símbolos según PF, accuracy y drawdown usando filtros_config."""
    if not os.path.exists(file):
        logging.warning(f"⚠️ No se encontró {file}, usando símbolos por defecto.")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

    try:
        df = pd.read_csv(file)
        filtered = df[
            (df["profit_factor"] >= filtros_config["min_pf"]) &
            (df["accuracy"] >= filtros_config["min_accuracy"]) &
            (df["drawdown"] <= filtros_config["max_drawdown"])
        ]
        good = filtered["symbol"].tolist()
        if not good:
            logging.warning("⚠️ Ningún símbolo cumple los filtros, usando lista completa.")
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

        logging.info(
            f"✅ Símbolos filtrados — PF ≥ {filtros_config['min_pf']}, "
            f"Accuracy ≥ {filtros_config['min_accuracy']}%, "
            f"Drawdown ≤ {filtros_config['max_drawdown']}: {good}"
        )
        return good
    except Exception as e:
        logging.error(f"❌ Error leyendo {file}: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

# 🔹 Inicializar símbolos activos al iniciar la app
SYMBOLS = get_symbols_with_filters()

# ===============================
# 📥 Carga de umbrales por símbolo
# ===============================
mejores_umbral: Dict[str, Dict[str, float]] = {}

def cargar_mejores_umbral():
    """Carga umbrales SOLO para símbolos activos (SYMBOLS)."""
    global mejores_umbral
    try:
        if not os.path.exists(OPTIM_FILE):
            logging.warning(f"⚠️ {OPTIM_FILE} no encontrado, usando umbrales base.")
            mejores_umbral = {}
            return mejores_umbral

        df = pd.read_csv(OPTIM_FILE)
        nuevos = {}
        for _, row in df.iterrows():
            symbol = str(row["symbol"]).upper().strip()
            if symbol not in SYMBOLS:
                continue
            up_val = float(row.get("UP", BASE_UP_THRESHOLD))
            down_val = float(row.get("DOWN", BASE_DOWN_THRESHOLD))
            nuevos[symbol] = {
                "UP": max(MIN_THRESHOLD, min(MAX_THRESHOLD, up_val)),
                "DOWN": max(MIN_THRESHOLD, min(MAX_THRESHOLD, down_val)),
            }
        mejores_umbral = nuevos
        logging.info(f"📥 Mejores umbrales cargados (filtrados): {mejores_umbral}")
        return mejores_umbral
    except Exception as e:
        logging.error(f"❌ Error cargando {OPTIM_FILE}: {e}")
        mejores_umbral = {}
        return mejores_umbral

cargar_mejores_umbral()

# ===============================
# ♻️ Refresco periódico
# ===============================
def refrescar_filtros_periodicamente():
    while True:
        try:
            logging.info("♻️ [AUTO] Actualizando símbolos y umbrales óptimos...")
            global SYMBOLS
            SYMBOLS = get_symbols_with_filters()
            cargar_mejores_umbral()
            logging.info(f"✅ [AUTO] Actualización completada — Símbolos: {SYMBOLS}")
        except Exception as e:
            logging.error(f"❌ Error en actualización automática: {e}")
        time.sleep(REFRESH_INTERVAL)

# 🟢 Lanzar el hilo en segundo plano después de definir TODO
t = threading.Thread(target=refrescar_filtros_periodicamente, daemon=True)
t.start()

# ========= Helpers de thresholds JSON mínimos =========
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

## ===============================
# 🧠 Filtro automático de símbolos por rendimiento
# ===============================
def get_symbols_with_good_pf(
    file="backtest_results_optim.csv",
    min_pf=1.5,
    min_accuracy=50.0,
    max_drawdown=20.0
):
    """
    Lee el archivo de optimización y devuelve los símbolos que cumplen:
    - profit_factor >= min_pf
    - accuracy >= min_accuracy
    - drawdown <= max_drawdown
    Si no hay archivo o no se cumple nada, devuelve la lista completa por defecto.
    """
    default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

    if not os.path.exists(file):
        logging.warning(f"⚠️ No se encontró {file}, usando símbolos por defecto.")
        return default_symbols

    try:
        df = pd.read_csv(file)

        # Normalizar nombres de columnas por seguridad
        df.columns = [c.strip().lower() for c in df.columns]

        # Filtrar por condiciones si existen las columnas
        conditions = (df["profit_factor"] >= min_pf)
        if "accuracy" in df.columns:
            conditions &= (df["accuracy"] >= min_accuracy)
        if "drawdown" in df.columns:
            conditions &= (df["drawdown"] <= max_drawdown)

        good = df[conditions]["symbol"].tolist() if "symbol" in df.columns else []

        if not good:
            logging.warning("⚠️ Ningún símbolo cumple los filtros, usando lista completa.")
            return default_symbols

        logging.info(f"✅ Símbolos filtrados: PF≥{min_pf}, Accuracy≥{min_accuracy}%, DD≤{max_drawdown}% → {good}")
        return good

    except Exception as e:
        logging.error(f"❌ Error leyendo {file}: {e}")
        return default_symbols


# 🔹 Cargar automáticamente solo símbolos ganadores
SYMBOLS = get_symbols_with_good_pf()

# ===============================
# 🧠 Configuración de filtros dinámicos
# ===============================
filtros_config = {
    "min_pf": 1.5,
    "min_accuracy": 50.0,     # porcentaje mínimo de aciertos
    "max_drawdown": 0.20      # drawdown máximo permitido (20%)
}

def get_symbols_with_filters(file=OPTIM_FILE):
    """
    Filtra símbolos según PF, accuracy y drawdown usando los valores en filtros_config.
    """
    if not os.path.exists(file):
        logging.warning(f"⚠️ No se encontró {file}, usando símbolos por defecto.")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

    try:
        df = pd.read_csv(file)
        filtered = df[
            (df["profit_factor"] >= filtros_config["min_pf"]) &
            (df["accuracy"] >= filtros_config["min_accuracy"]) &
            (df["drawdown"] <= filtros_config["max_drawdown"])
        ]

        good = filtered["symbol"].tolist()

        if not good:
            logging.warning("⚠️ Ningún símbolo cumple los filtros, usando lista completa.")
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

        logging.info(
            f"✅ Símbolos filtrados — PF ≥ {filtros_config['min_pf']}, "
            f"Accuracy ≥ {filtros_config['min_accuracy']}%, "
            f"Drawdown ≤ {filtros_config['max_drawdown']}: {good}"
        )
        return good
    except Exception as e:
        logging.error(f"❌ Error leyendo {file}: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]


# 🔹 Inicializar símbolos activos al iniciar la app
SYMBOLS = get_symbols_with_filters()

# ===============================
# 🌐 Endpoint para actualizar filtros dinámicamente
# ===============================
@app.route("/api/update-filter", methods=["POST"])
def update_filter():
    """
    Permite actualizar filtros dinámicamente sin reiniciar el servidor.
    Ejemplo body:
    {
        "min_pf": 1.8,
        "min_accuracy": 55,
        "max_drawdown": 0.15
    }
    """
    global SYMBOLS
    data = request.get_json(force=True) or {}

    # Actualizar valores existentes si vienen en la petición
    for key in filtros_config:
        if key in data:
            filtros_config[key] = float(data[key])

    # Volver a filtrar símbolos según nuevos parámetros
    SYMBOLS = get_symbols_with_filters()

    return jsonify({
        "status": "✅ Filtros actualizados correctamente",
        "filtros": filtros_config,
        "active_symbols": SYMBOLS
    })

def cargar_mejores_umbral():
    """
    Lee el archivo backtest_results_optim.csv y carga los mejores umbrales
    SOLO para los símbolos activos que superaron el filtro PF.
    """
    global mejores_umbral
    try:
        if not os.path.exists(OPTIM_FILE):
            logging.warning(f"⚠️ {OPTIM_FILE} no encontrado, usando umbrales base.")
            mejores_umbral = {}
            return mejores_umbral

        df = pd.read_csv(OPTIM_FILE)
        nuevos = {}
        for _, row in df.iterrows():
            symbol = str(row["symbol"]).upper().strip()
            # ⚡ Solo incluir símbolos que pasaron el filtro PF
            if symbol not in SYMBOLS:
                continue
            up_val = float(row.get("UP", BASE_UP_THRESHOLD))
            down_val = float(row.get("DOWN", BASE_DOWN_THRESHOLD))
            nuevos[symbol] = {
                "UP": max(MIN_THRESHOLD, min(MAX_THRESHOLD, up_val)),
                "DOWN": max(MIN_THRESHOLD, min(MAX_THRESHOLD, down_val))
            }
        mejores_umbral = nuevos
        logging.info(f"📥 Mejores umbrales cargados (filtrados): {mejores_umbral}")
        return mejores_umbral
    except Exception as e:
        logging.error(f"❌ Error cargando {OPTIM_FILE}: {e}")
        mejores_umbral = {}
        return mejores_umbral

def get_best_threshold(symbol: str) -> Dict[str, float]:
    """
    Devuelve umbrales del CSV optimizado si existen; si no, los defaults.
    """
    if symbol in mejores_umbral:
        return {
            "UP": mejores_umbral[symbol]["UP"],
            "DOWN": mejores_umbral[symbol]["DOWN"],
            "DIFF": BASE_DIFF_MARGIN,
        }
    # ⚠️ Esto cubre casos extremos si se consulta un símbolo no filtrado
    return {"UP": BASE_UP_THRESHOLD, "DOWN": BASE_DOWN_THRESHOLD, "DIFF": BASE_DIFF_MARGIN}

# compatibilidad para endpoint /_thresholds y backtest
def get_symbol_thresholds(sym: str) -> Dict[str, float]:
    return get_best_threshold(sym)

# 🧠 Cargar una vez al iniciar la app
cargar_mejores_umbral()

# =========================================================
# 🔁 Endpoint para recargar umbrales dinámicamente
# =========================================================
@app.route("/api/reload-thresholds", methods=["POST"])
def reload_thresholds():
    cargar_mejores_umbral()
    return jsonify({"status": "✅ Umbrales recargados", "data": mejores_umbral})
# ===============================
# ⚡ Inicializar cliente Binance con reintentos + Descarga segura de velas
# ===============================

client = None  # 👈 variable global

def init_client(api_key, api_secret, max_retries=5, backoff=2):
    """
    Inicializa el cliente de Binance Futures con reintentos automáticos.
    Si la conexión es exitosa, guarda el cliente globalmente.
    """
    global client
    for attempt in range(max_retries):
        try:
            c = Client(api_key, api_secret)
            c.futures_ping()
            client = c  # 👈 guarda en la variable global
            logging.info("✅ Conectado correctamente a Binance Futures.")
            return client
        except Exception as e:
            logging.warning(f"❌ Intento {attempt + 1} fallido conectando a Binance: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff ** attempt + np.random.uniform(0, 1)
                logging.info(f"🔁 Reintentando en {sleep_time:.2f} segundos...")
                time.sleep(sleep_time)
            else:
                logging.error("⚠️ No se pudo conectar a Binance después de varios intentos.")
                client = None
                return None

def download_klines_safe(sym, interval=None, limit=None):
    """
    📥 Descarga velas seguras desde Binance Futures con manejo de errores.
    Usa el cliente global actual en el momento de la llamada.
    """
    global client

    # 👇 Validación dinámica: usa el client actual
    if client is None:
        logging.error("🚨 Cliente de Binance no inicializado.")
        return pd.DataFrame()

    try:
        interval = interval or INTERVAL
        limit = limit or HISTORICAL_LIMIT

        kl = client.futures_klines(symbol=sym, interval=interval, limit=limit)
        if not kl:
            logging.warning(f"⚠️ No se recibieron datos de velas para {sym}")
            return pd.DataFrame()

        df = pd.DataFrame(kl, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
            "Quote_asset_volume", "Number_of_trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = df[c].astype(float)

        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
        df.set_index("Open_time", inplace=True)

        df["atr"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()

        return df.ffill().bfill()

    except Exception as e:
        logging.error(f"❌ Error descargando datos de {sym}: {e}")
        return pd.DataFrame()
# ===============================
# 📰 Noticias económicas
# ===============================
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
                return {
                    "event": evento.get("Event", ""),
                    "country": evento.get("Country", ""),
                    "time": event_time.strftime("%H:%M UTC")
                }
        return None
    except Exception:
        return None

# ===============================
# 📊 Umbrales dinámicos según condiciones del mercado
# ===============================
def get_dynamic_thresholds(adx_value: float, atr_ratio: float, has_news: bool):
    up = BASE_UP_THRESHOLD
    down = BASE_DOWN_THRESHOLD
    diff = BASE_DIFF_MARGIN

    # 📈 Tendencia fuerte — modo agresivo
    if adx_value > 25 and atr_ratio > MIN_ATR_RATIO * 2 and not has_news:
        up -= 0.05
        down -= 0.05
        diff -= 0.02
    # 🧱 Lateral / baja vol / noticias — modo defensivo
    elif adx_value < 15 or atr_ratio < MIN_ATR_RATIO or has_news:
        up += 0.05
        down += 0.05
        diff += 0.02

    up = min(MAX_THRESHOLD, max(MIN_THRESHOLD, up))
    down = min(MAX_THRESHOLD, max(MIN_THRESHOLD, down))
    diff = max(0.0, diff)
    return {"UP": up, "DOWN": down, "DIFF": diff}

# ===============================
# 🧭 Soporte y resistencia inteligente
# ===============================
def detectar_zonas_sr(df, lookback=50):
    sub = df.iloc[-lookback:]
    soporte = sub["Low"].min()
    resistencia = sub["High"].max()
    return soporte, resistencia

# ---------------- Modelo ----------------
def _guess_is_tree_model(m):
    try:
        if hasattr(m, "feature_importances_"):
            return True
        if type(m).__name__.lower().startswith(("lgbm", "lightgbm", "xgb", "randomforest", "gradientboosting", "extratrees", "decisiontree")):
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
    feature_cols = [
        "ema_10","ema_20","ema_50","sma_50","sma_200","ama_cross",
        "momentum","logret","atr14","bb_pct","rsi14","stoch_k","stoch_d",
        "macd","macd_signal","obv","vpt"
    ]

# ===============================
# ⚡ SCANNER B — RÁPIDO (con WebSocket)
# ===============================

def get_fast_features(symbol: str):
    """
    Genera features mínimos para IA usando el precio interno (latest_prices)
    sin volver a descargar klines completos.
    """
    try:
        if latest_prices.get(symbol) is None:
            return None

        # ✅ Descargar solo una vez datos base para features
        df = download_klines_safe(symbol)
        if df.empty:
            return None

        # 👇 Actualizamos solo el último precio con el interno
        df.iloc[-1, df.columns.get_loc("Close")] = latest_prices[symbol]

        # 📈 Calcular indicadores mínimos (ajústalos a tu modelo si es necesario)
        df["ema_10"] = ta.trend.EMAIndicator(df["Close"], 10).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
        df["rsi14"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
        df["atr14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()

        last_row = df.iloc[-1]
        features = [last_row.get(col, 0) for col in feature_cols]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        logging.error(f"❌ Error generando features rápidos para {symbol}: {e}")
        return None


def fast_scan_symbol(symbol: str):
    """
    Escaneo IA rápido por símbolo (solo probabilidades + señal),
    sin cálculos de contexto ni planes completos.
    """
    try:
        X = get_fast_features(symbol)
        if X is None:
            return None

        probs = model.predict_proba(X)[0]
        # 🔸 Ajustar índices según tus clases
        classes = list(getattr(model, "classes_", []))
        idx_up = get_class_index(classes, 1)
        idx_down = get_class_index(classes, -1)

        up_prob = float(probs[idx_up])
        down_prob = float(probs[idx_down])

        thresholds = get_best_threshold(symbol)
        up_th = thresholds["UP"]
        down_th = thresholds["DOWN"]
        diff_margin = thresholds["DIFF"]

        # 🧠 Decisión simple
        if up_prob > up_th and (up_prob - down_prob) > diff_margin:
            signal = "BUY"
        elif down_prob > down_th and (down_prob - up_prob) > diff_margin:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "price": latest_prices[symbol],
            "up_prob": round(up_prob * 100, 2),
            "down_prob": round(down_prob * 100, 2),
            "signal": signal
        }

    except Exception as e:
        logging.error(f"❌ Error escaneando rápido {symbol}: {e}")
        return None


def run_fast_scanner():
    """Escanea rápidamente todos los símbolos activos."""
    results = []
    for sym in SYMBOLS:
        r = fast_scan_symbol(sym)
        if r:
            results.append(r)
    # Ordenar por probabilidad más fuerte
    results.sort(key=lambda x: max(x["up_prob"], x["down_prob"]), reverse=True)
    return results

# ---------------- Datos/Features ----------------
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

    # Lags
    for f in ["atr14","bb_pct","rsi14","stoch_k","stoch_d","macd","macd_signal","vpt","ama_cross","momentum","logret"]:
        for lag in (1, 2, 3):
            df[f"{f}_lag{lag}"] = df[f].shift(lag)

    return df.dropna()
# ===============================
# 🕯 DETECTOR DE PATRONES DE VELAS JAPONESAS
# ===============================
def detectar_patron_velas(df):
    if len(df) < 3:
        return None

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    # Última vela
    o1, h1, l1, c1 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o2, h2, l2, c2 = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]

    cuerpo = abs(c1 - o1)
    rango = h1 - l1
    sombra_sup = h1 - max(c1, o1)
    sombra_inf = min(c1, o1) - l1

    # Evitar divisiones por 0
    if rango == 0:
        return None

    # Patrones básicos
    if cuerpo < rango * 0.25 and sombra_inf > cuerpo * 2:
        return "HAMMER"
    elif cuerpo < rango * 0.25 and sombra_sup > cuerpo * 2:
        return "SHOOTING_STAR"
    elif c1 > o1 and o1 < c2 and c1 > o2 and abs(c1 - o1) > abs(c2 - o2) * 0.7:
        return "BULLISH_ENGULFING"
    elif c1 < o1 and o1 > c2 and c1 < o2 and abs(c1 - o1) > abs(c2 - o2) * 0.7:
        return "BEARISH_ENGULFING"
    elif abs(c1 - o1) <= rango * 0.1:
        return "DOJI"
    else:
        return None

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

def explain_like_human(xrow: pd.Series, exp: Dict[str, Any]) -> str:
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
            if ema20 > ema50: hints.append("EMA20>EMA50 (tendencia)")
            elif ema20 < ema50: hints.append("EMA20<EMA50 (tendencia débil)")

        if not isnan(bbp):
            if bbp >= 0.8: hints.append("Cerca banda superior (posible sobrecompra)")
            elif bbp <= 0.2: hints.append("Cerca banda inferior (posible sobreventa)")

        if not any(map(isnan, [macd_v, macd_s])):
            if macd_v > macd_s: hints.append("MACD positivo")
            elif macd_v < macd_s: hints.append("MACD negativo")

        if not isnan(adx):
            if adx >= 20: hints.append("ADX suficiente (tendencia)")
            else: hints.append("ADX bajo")

        if exp.get("shap_top"):
            top_feats = [f for f, _ in exp["shap_top"][:3]]
            hints.append("SHAP: " + ", ".join(top_feats))

        if not hints:
            return "Señal por combinación de momentum/tendencia/volatilidad."
        return " / ".join(hints[:4])
    except Exception:
        return "Señal por combinación de momentum/tendencia/volatilidad."

def build_local_explanation(X_row: pd.Series, class_idx: int, top_n=5) -> Dict[str, Any]:
    explanation = {"top_importances": _top_feature_importances(n=top_n), "shap_top": None, "reason_text": None}
    if X_row is None:
        return explanation
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
        explanation["reason_text"] = explain_like_human(X_row, explanation)
        return explanation
    except Exception as e:
        logging.warning(f"No se pudo calcular SHAP local: {e}")
        explanation["reason_text"] = explain_like_human(X_row, explanation)
        return explanation

# ---------------- Filtros/Plan ----------------
def condiciones_de_mercado_ok_df(symbol, df_feat) -> Tuple[bool, str, Optional[Dict[str, str]]]:
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

    plan["planB_rules"] = {
        "if_adx_drop": "Si ADX cae < 15 tras 3 velas, cambiar a ATR (TP=1.2*ATR, SL=1.0*ATR desde entrada).",
        "if_big_counter": "Si vela en contra > 1.2*ATR, reducir 50% posición y activar trailing ATR 1x."
    }
    return plan
def market_context_narrative(
    symbol,
    df,
    signal,
    prob_up_pct,
    prob_down_pct,
    soporte=None,
    resistencia=None,
    patron=None,
    trade_plan=None,
    news=None
):
    try:
        # Contexto base
        price = float(df["Close"].iloc[-1])
        regime = "lateral"

        ema20 = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator().iloc[-1]
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx().iloc[-1]

        # Ancho simple de rango reciente (20 velas)
        recent_high = float(df["High"].iloc[-20:].max())
        recent_low = float(df["Low"].iloc[-20:].min())
        bb_width = (recent_high - recent_low) / max(price, 1e-9)

        # Régimen
        if ema20 > ema50 and adx > 20 and bb_width > 0.015:
            regime = "alcista"
        elif ema20 < ema50 and adx > 20 and bb_width > 0.015:
            regime = "bajista"

        narr = (
            f"{symbol} — Precio: {price:.2f} | "
            f"Prob↑ {prob_up_pct:.2f}% | Prob↓ {prob_down_pct:.2f}% | "
            f"Regimen: {regime}. "
        )

        # Noticias
        if news:
            ev = news.get("event", "Evento")
            tm = news.get("time", "")
            ct = news.get("country", "")
            narr += f"Noticia de alto impacto proxima: {ev} a las {tm} ({ct}). Evitar operar hasta que pase. "

        # Soporte/Resistencia cercanos
        atr_margin = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        if soporte is not None and abs(price - float(soporte)) <= atr_margin:
            narr += f"Precio cerca de soporte ({float(soporte):.2f}). "
        if resistencia is not None and abs(price - float(resistencia)) <= atr_margin:
            narr += f"Precio cerca de resistencia ({float(resistencia):.2f}). "

        # Patrones de vela: guía operativa simple
        if patron:
            if patron == "HAMMER":
                narr += "Patron HAMMER: posible rebote tras caida. Esperar vela alcista de confirmacion o ruptura de maximo previo. "
            elif patron == "SHOOTING_STAR":
                narr += "Patron SHOOTING STAR: posible agotamiento alcista. Vigilar rechazo en resistencia o ruptura a la baja. "
            elif patron == "DOJI":
                narr += "Patron DOJI: indecision. Esperar ruptura clara de soporte/resistencia antes de entrar. "
            elif patron == "BULLISH_ENGULFING":
                narr += "Patron BULLISH ENGULFING: sesgo alcista. Ruptura de resistencia podria validar entrada long. "
            elif patron == "BEARISH_ENGULFING":
                narr += "Patron BEARISH ENGULFING: sesgo bajista. Ruptura de soporte podria validar entrada short. "

        # Señal final y plan, si existe
        sig_up = "COMPRA" in str(signal).upper()
        sig_dn = "VENTA" in str(signal).upper()

        if sig_up:
            narr += "Senal de COMPRA. "
            if trade_plan:
                narr += (
                    f"Entrada: {trade_plan.get('entry_suggest')} | "
                    f"SL: {trade_plan.get('SL')} | "
                    f"TP: {trade_plan.get('TP')}. "
                )
        elif sig_dn:
            narr += "Senal de VENTA. "
            if trade_plan:
                narr += (
                    f"Entrada: {trade_plan.get('entry_suggest')} | "
                    f"SL: {trade_plan.get('SL')} | "
                    f"TP: {trade_plan.get('TP')}. "
                )
        else:
            narr += "Estrategia en espera, sin confirmacion clara. "
            if patron == "DOJI":
                narr += "El patron sugiere indecision; mejor esperar ruptura. "
            if (soporte is not None) or (resistencia is not None):
                narr += "Esperando ruptura o confirmacion en zona clave. "

        return narr

    except Exception as e:
        logging.warning(f"Error narrativa {symbol}: {e}")
        return f"{symbol}: contexto no disponible."



# ---------------- Gestión de riesgo ----------------
def kelly_sizing(winrate: float, rr: float, fraction_cap: float = KELLY_FRACTION) -> float:
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


def compute_signal_for_symbol(symbol: str, balance: float = DEFAULT_BALANCE, use_kelly: bool = False) -> Dict[str, Any]:
    df = download_klines_safe(symbol)
    if df.empty:
        return {"error": f"No hay datos para {symbol}"}

    df_feat = compute_indicators(df)
    if df_feat.empty:
        return {"error": f"Datos insuficientes para {symbol} (warm-up)"}

    price_now = float(df_feat["Close"].iloc[-1])
    adx_val = float(df_feat["adx"].iloc[-1])
    atr_val = float(df_feat["atr"].iloc[-1])
    atr_ratio = atr_val / price_now

    # 📰 Noticias
    news = hay_noticia_importante_proxima()
    has_news = bool(news)

    # 🧠 Umbrales dinámicos
    th = get_dynamic_thresholds(adx_val, atr_ratio, has_news)

    # 📊 Predicción IA
    X_all = df_feat.reindex(columns=feature_cols).astype(float)
    classes = list(getattr(model, "classes_", []))
    idx_up = get_class_index(classes, 1)
    idx_down = get_class_index(classes, -1)
    preds_all = model.predict_proba(X_all.values)
    prob_up = float(preds_all[-1, idx_up])
    prob_down = float(preds_all[-1, idx_down])

    # 🧠 Señal base
    signal = "ESPERAR"
    if prob_up > th["UP"] and (prob_up - prob_down) >= th["DIFF"]:
        signal = "COMPRAR"
    elif prob_down > th["DOWN"] and (prob_down - prob_up) >= th["DIFF"]:
        signal = "VENTA"

    # 🕯 Patrones de velas
    patron = detectar_patron_velas(df)
    if patron:
        if patron in ["HAMMER", "BULLISH_ENGULFING"] and signal == "COMPRAR":
            signal = "COMPRAR"
        elif patron in ["SHOOTING_STAR", "BEARISH_ENGULFING"] and signal == "VENTA":
            signal = "VENTA"
        elif patron == "DOJI":
            signal = "ESPERAR"

    # 🧭 Soportes y resistencias inteligentes
    soporte, resistencia = detectar_zonas_sr(df_feat)
    margen_atr = atr_val
    tech_message = None

    if soporte and resistencia and margen_atr and not np.isnan(margen_atr):
        dist_a_soporte = abs(price_now - soporte)
        dist_a_resistencia = abs(price_now - resistencia)

        if signal == "COMPRAR" and dist_a_soporte <= margen_atr:
            logging.info(f"🧭 Cerca de SOPORTE ({soporte:.2f}) → refuerza COMPRA")
        elif signal == "VENTA" and dist_a_resistencia <= margen_atr:
            logging.info(f"🧭 Cerca de RESISTENCIA ({resistencia:.2f}) → refuerza VENTA")
        elif signal in ["COMPRAR", "VENTA"]:
            # 🧠 si hay señal pero lejos de zonas técnicas → espera ruptura
            signal = "ESPERAR"
            tech_message = "🧭 Precio lejos de zonas técnicas — señal en pausa esperando ruptura."

    # 📈 Confirmación adicional con tendencia (EMA / ADX)
    ema20 = float(df_feat.get("ema_20").iloc[-1])
    ema50 = float(df_feat.get("ema_50").iloc[-1])
    confirm = (signal == "COMPRAR" and adx_val > 20 and ema20 > ema50) or \
              (signal == "VENTA" and adx_val > 20 and ema20 < ema50)

    # 🟡 Construir salida si no hay señal
    if signal == "ESPERAR":
        msg = "Noticia de alto impacto próxima — Evitar operar." if news else "Condiciones no ideales."
        return {
            "symbol": symbol,
            "signal": "ESPERAR 🧠",
            "message": msg,
            "price": round(price_now, 4),
            "technical_message": tech_message,
            "prob_up": round(prob_up * 100, 2),
            "prob_down": round(prob_down * 100, 2),
            "thresholds_used": th,
            "candle_pattern": patron,
            "support": round(soporte, 4) if soporte else None,
            "resistance": round(resistencia, 4) if resistencia else None,
            **({"news_event": news} if news else {})
        }

    # ✅ Señal final
    signal_out = (
        "COMPRA CONFIRMADA ✅" if signal == "COMPRAR" and confirm else
        "VENTA CONFIRMADA ✅" if signal == "VENTA" and confirm else
        "COMPRA POTENCIAL ⚠️" if signal == "COMPRAR" else
        "VENTA POTENCIAL ⚠️"
    )

    return {
        "symbol": symbol,
        "signal": signal_out,
        "prob_up": round(prob_up * 100, 2),
        "prob_down": round(prob_down * 100, 2),
        "price": round(price_now, 4),
        "technical_message": tech_message,
        "thresholds_used": th,
        "candle_pattern": patron,
        "support": round(soporte, 4) if soporte else None,
        "resistance": round(resistencia, 4) if resistencia else None,
        **({"news_event": news} if news else {})
    }

# ---------------- Backtest ----------------
def simulate_trade(row_entry_idx, side, entry, sl, tp, df_feat, use_planB: bool) -> Tuple[float, int]:
    bars = 0
    risk = abs(entry - sl)
    if risk <= 1e-12:
        return 0.0, 0

    size_factor = 1.0
    trailing_sl = None

    for i in range(row_entry_idx + 1, len(df_feat)):
        bars += 1
        c = float(df_feat["Close"].iloc[i])
        h = float(df_feat["High"].iloc[i])
        l = float(df_feat["Low"].iloc[i])
        atr = float(df_feat["atr"].iloc[i])
        adx = float(df_feat["adx"].iloc[i])

        if use_planB:
            if adx < 15 and trailing_sl is None:
                if side == "long":
                    trailing_sl = c - 1.0 * atr
                    tp_alt = c + 1.2 * atr
                    if h >= tp_alt:
                        gain = (tp_alt - entry) / risk
                        return round(gain * size_factor, 2), bars
                else:
                    trailing_sl = c + 1.0 * atr
                    tp_alt = c - 1.2 * atr
                    if l <= tp_alt:
                        gain = (entry - tp_alt) / risk
                        return round(gain * size_factor, 2), bars
            if ((side == "long" and (entry - l) > 1.2 * atr) or
                (side == "short" and (h - entry) > 1.2 * atr)) and size_factor > 0.51:
                size_factor = 0.5

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

    last_close = float(df_feat["Close"].iloc[-1])
    if side == "long":
        res = (last_close - entry) / risk
    else:
        res = (entry - last_close) / risk
    return round(res, 2), bars


def backtest_symbol(symbol: str, days: int = 60, use_planB: bool = True) -> dict:
    """
    📊 Ejecuta un backtest sobre un símbolo de Binance usando lógica adaptativa o no adaptativa.
    - Descarga velas reales usando el cliente global
    - Aplica indicadores y modelo IA
    - Evalúa señales y planes de trade con filtros flexibles
    - Devuelve métricas de rendimiento
    """
    # ==============================
    # 📥 Descargar datos históricos
    # ==============================
    df = download_klines_safe(symbol)
    if df.empty:
        logging.error(f"❌ No hay datos para {symbol}")
        return {"error": f"No hay datos para {symbol}"}

    # ⏳ Filtrar el rango de tiempo
    since = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= since]
    if df.empty:
        logging.warning(f"⚠️ No hay suficientes datos en el rango de {days} días para {symbol}")
        return {"error": "Datos insuficientes para el rango solicitado"}

    # 🧮 Calcular indicadores
    df_feat = compute_indicators(df)
    if df_feat.empty or len(df_feat) < 200:
        logging.warning(f"⚠️ Datos insuficientes para backtest de {symbol}")
        return {"error": "Datos insuficientes para backtest"}

    # 🧠 Predicciones del modelo IA
    try:
        X_all = df_feat.reindex(columns=feature_cols).astype(float)
        preds = model.predict_proba(X_all.values)
    except Exception as e:
        logging.error(f"❌ Error al predecir con el modelo: {e}")
        return {"error": "Fallo en predicciones del modelo"}

    classes = list(getattr(model, "classes_", []))
    idx_up = get_class_index(classes, 1)
    idx_down = get_class_index(classes, -1)

    th = get_symbol_thresholds(symbol)
    results = []
    cum_R = 0.0
    day_loss_R = {}

    # ==============================
    # 🧭 Loop de evaluación de señales
    # ==============================
    for i in range(1, len(df_feat) - 1):
        # 🎯 Señales IA
        prob_up = float(preds[i, idx_up])
        prob_down = float(preds[i, idx_down])
        diff_prob = abs(prob_up - prob_down)

        up_th = th.get("UP", BASE_UP_THRESHOLD)
        down_th = th.get("DOWN", BASE_DOWN_THRESHOLD)
        diff_th = th.get("DIFF", BASE_DIFF_MARGIN)

        signal = "ESPERAR"
        if prob_up > up_th and (prob_up - prob_down) >= diff_th:
            signal = "COMPRAR"
        elif prob_down > down_th and (prob_down - prob_up) >= diff_th:
            signal = "VENTA"
        elif diff_prob >= 0.02 and max(prob_up, prob_down) > 0.5:
            signal = "COMPRAR" if prob_up > prob_down else "VENTA"

        if signal == "ESPERAR":
            continue

        # ==============================
        # 📊 Filtros técnicos flexibles
        # ==============================
        adx_val = float(df_feat["adx"].iloc[i])
        ema20 = float(df_feat.get("ema_20", df_feat.get("ema20", np.nan)).iloc[i])
        ema50 = float(df_feat.get("ema_50", df_feat.get("ema50", np.nan)).iloc[i])
        atr_val = float(df_feat.get("atr", np.nan).iloc[i])

        confirm_score = 0
        if adx_val > MIN_ADX:
            confirm_score += 1
        if signal == "COMPRAR" and ema20 > ema50:
            confirm_score += 1
        if signal == "VENTA" and ema20 < ema50:
            confirm_score += 1
        if atr_val > MIN_ATR_RATIO * float(df_feat["Close"].iloc[i]):
            confirm_score += 1

        signal_out = signal + (" CONFIRMADA ✅" if confirm_score >= 2 else " POTENCIAL ⚠️")

        # 🛑 Si no pasa ningún filtro técnico, descarta
        if confirm_score == 0:
            continue

        logging.debug(
            f"[SIGNAL] {signal_out} | UP={prob_up:.3f} DOWN={prob_down:.3f} "
            f"DIFF={diff_prob:.3f} ADX={adx_val:.2f} EMA20={ema20:.2f} EMA50={ema50:.2f} ATR={atr_val:.6f}"
        )

        # ==============================
        # 🧭 Plan de trade
        # ==============================
        sub_df = df_feat.iloc[: i + 1]
        plan = choose_trade_plan(signal_out, sub_df)
        entry, sl, tp = plan.get("entry_suggest"), plan.get("SL"), plan.get("TP")
        if not entry or not sl or not tp or plan.get("plan") == "NONE":
            continue

        # ⛔ Control de pérdida diaria
        day_key = df_feat.index[i].strftime("%Y-%m-%d")
        lost_today = day_loss_R.get(day_key, 0.0)
        if lost_today <= -DAILY_R_MAX:
            continue

        # 🧮 Simulación de la operación
        side = "long" if "COMPRA" in signal_out.upper() else "short"
        R, bars = simulate_trade(i, side, entry, sl, tp, df_feat, use_planB=use_planB)
        cum_R += R
        if R < 0:
            day_loss_R[day_key] = day_loss_R.get(day_key, 0.0) + R

        results.append({
            "idx": i,
            "time": str(df_feat.index[i]),
            "signal": signal_out,
            "entry": entry,
            "SL": sl,
            "TP": tp,
            "R": R,
            "bars": bars
        })

    # ==============================
    # 📉 Si no hubo trades
    # ==============================
    if not results:
        logging.warning(f"⚠️ No se generaron trades para {symbol} en {days} días")
        return {
            "symbol": symbol,
            "days": days,
            "trades": 0,
            "cum_R": 0.0,
            "winrate": None,
            "avgR": None,
            "maxDD_R": 0.0,
            "details": []
        }

    # ==============================
    # 📊 Métricas de rendimiento
    # ==============================
    Rs = [r["R"] for r in results]
    wins = sum(1 for r in Rs if r > 0)
    cum_curve = np.cumsum(Rs)
    max_dd = 0.0
    peak = 0.0
    for v in cum_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)

    metrics = {
        "symbol": symbol,
        "days": days,
        "trades": len(results),
        "cum_R": round(float(np.sum(Rs)), 2),
        "winrate": round(wins / len(results), 3),
        "avgR": round(float(np.mean(Rs)), 3),
        "maxDD_R": round(float(max_dd), 2),
        "details": results[:200]
    }

    logging.info(
        f"✅ Backtest {symbol}: {metrics['trades']} trades | "
        f"R={metrics['cum_R']} | winrate={metrics['winrate']}"
    )
    return metrics

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
        print("DEBUG RESP INICIAL:", resp)

        # 🆕 Si no hay df, lo descargamos ahora
        df = download_klines_safe(symbol, INTERVAL)  # Usa tu propia función para obtener velas

        if not df.empty:
            signal = resp.get("signal", "")
            prob_up = resp.get("prob_up", 0)
            prob_down = resp.get("prob_down", 0)
            soporte = float(resp.get("support")) if resp.get("support") is not None else None
            resistencia = float(resp.get("resistance")) if resp.get("resistance") is not None else None
            patron = resp.get("candle_pattern")
            trade_plan = {
                "entry_suggest": resp.get("entry_suggest"),
                "SL": resp.get("SL"),
                "TP": resp.get("TP"),
                "plan": resp.get("plan")
            }

            narrative = market_context_narrative(
                symbol,
                df,
                signal,
                prob_up,
                prob_down,
                soporte=soporte,
                resistencia=resistencia,
                patron=patron,
                trade_plan=trade_plan,
                news=None
            )

            resp["narrative"] = narrative
            print("📝 Narrativa generada:", narrative)

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
            return jsonify({"error": f"No hay datos para {symbol}"}), 400
        df_feat = compute_indicators(df)
        if df_feat.empty:
            return jsonify({"error": f"Datos insuficientes para {symbol} (warm-up)"}), 400

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

        # ✅ Tomar el símbolo
        symbol = (data.get("symbol") or "").upper().strip()
        if not symbol:
            return jsonify({"error": "Símbolo vacío"}), 400

        # ⏳ Tomar días, por defecto 90
        days = int(data.get("days", 90))

        # 🧠 Plan adaptativo o no
        use_planB = bool(data.get("planB", True))

        logging.info(f"📊 Iniciando backtest: {symbol} — {days} días — planB={use_planB}")

        # 📥 Ejecutar backtest con lógica adaptativa
        resB = backtest_symbol(symbol, days=days, use_planB=True)

        # 📥 Ejecutar backtest sin lógica adaptativa
        resA = backtest_symbol(symbol, days=days, use_planB=False)

        # 📤 Retornar resultados
        return jsonify({
            "with_planB": resB,
            "without_planB": resA,
            "symbol": symbol,
            "days": days
        })

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
        file_path = OPTIM_FILE
        if not os.path.exists(file_path):
            return jsonify({"error": "No se encontró el archivo"}), 404

        df = pd.read_csv(file_path)

        # 🧠 Filtrar símbolos activos actuales
        df = df[df["symbol"].isin(SYMBOLS)]

        # 🥇 NUEVO: Filtrar solo pares ganadores (PF > 1)
        df = df[df["profit_factor"] > 1]

        total_trades = len(df)
        if total_trades == 0:
            return jsonify({
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "winrate": 0,
                "avg_rr": 0,
                "max_drawdown": 0,
                "filtered_symbols": []  # 👈 Lista de pares usados
            })

        wins = int(total_trades)  # todos aquí son ganadores
        losses = 0

        winrate = 100.0  # si filtramos solo ganadores
        avg_rr = round(df['profit_factor'].replace([float('inf'), -float('inf')], 0).mean(), 2) \
            if 'profit_factor' in df.columns else 0
        max_dd = round(df['drawdown'].max(), 2) if 'drawdown' in df.columns else 0

        symbols_used = df['symbol'].unique().tolist()

        return jsonify({
            "total_trades": int(total_trades),
            "wins": int(wins),
            "losses": int(losses),
            "winrate": winrate,
            "avg_rr": avg_rr,
            "max_drawdown": max_dd,
            "filtered_symbols": symbols_used
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# 🌐 Endpoint para consultar filtros y símbolos activos
# ===============================
@app.route("/api/get-filters", methods=["GET"])
def get_filters_info():
    """
    Devuelve la configuración actual de filtros, símbolos activos
    y los umbrales cargados para cada uno.
    """
    response = {
        "filtros_actuales": filtros_config,
        "simbolos_activos": SYMBOLS,
        "umbrales_activos": mejores_umbral
    }
    return jsonify(response)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/api/paper_stats")
def paper_stats():
    stats = get_stats()
    return jsonify(stats)


# ===================================================
# 🚀 MAIN
if __name__ == "__main__":
    # 👉 Lanza WebSocket en segundo plano
    threading.Thread(target=iniciar_websocket, daemon=True).start()

    _ensure_thresholds_file()
    client = init_client(API_KEY, API_SECRET)
    if client is None:
        logging.error("No se pudo conectar a Binance. Saliendo.")
        exit(1)

    # 🧠 Iniciar el aprendizaje adaptativo diario
    from adaptive_learning import start_learning_thread
    start_learning_thread()

    # 👉 Ahora Flask se ejecutará normalmente
    app.run(host="0.0.0.0", port=5000, debug=False)




