# ===============================
# 📊 Backtest Diario + Optimización de Umbrales (Full Features) + Progreso
# ===============================

import os
import joblib
import pandas as pd
import numpy as np
import ta
import schedule
import time
import logging
from binance.client import Client
from datetime import datetime
from itertools import product

from pyparsing import results

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY") or "TU_API_KEY_REAL"
API_SECRET = os.getenv("API_SECRET") or "TU_API_SECRET_REAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
HISTORICAL_LIMIT = 1500
MODEL_FILE = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"
CAPITAL_INICIAL = 10000

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ---------------- FUNCIONES BASE ----------------
def init_client(api_key, api_secret):
    client = Client(api_key, api_secret)
    try:
        client.futures_ping()
        logging.info("✅ Conectado correctamente a Binance Futures.")
    except Exception as e:
        logging.warning(f"⚠️ No se pudo validar conexión con Binance: {e}")
    return client

def download_klines_safe(client, sym, interval=INTERVAL, limit=HISTORICAL_LIMIT):
    try:
        kl = client.futures_klines(symbol=sym, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
            "Quote_asset_volume", "Number_of_trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
        ])
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = df[c].astype(float)
        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
        df.set_index("Open_time", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error descargando datos de {sym}: {e}")
        return pd.DataFrame()

# ---------------- FEATURES ----------------
def compute_indicators(df):
    try:
        df["rsi14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["atr14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bb_pct"] = (df["Close"] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["vpt"] = ta.volume.VolumePriceTrendIndicator(df["Close"], df["Volume"]).volume_price_trend()
        df["momentum"] = df["Close"].diff()
        df["logret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["bb_width"] = df["High"] - df["Low"]
        df["ama_cross"] = np.sign(df["Close"].diff())

        # 👇 NUEVO: indicadores usados en la lógica Flask
        df["ema_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()

        return df
    except Exception as e:
        logging.error(f"Error calculando indicadores: {e}")
        return pd.DataFrame()


def add_lag_features(df, col, lags=3):
    for i in range(1, lags + 1):
        df[f"{col}_lag{i}"] = df[col].shift(i)
    return df

def build_features(df, sym, feature_cols):
    df = compute_indicators(df)
    if df.empty:
        return df

    for col in ["atr14", "bb_pct", "rsi14", "stoch_k", "stoch_d",
                "macd", "macd_signal", "vpt", "ama_cross", "momentum", "logret"]:
        if col in df.columns:
            df = add_lag_features(df, col, lags=3)

    for s in SYMBOLS:
        df[f"sym_{s}"] = 1 if s == sym else 0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df.ffill().bfill().dropna()
    return df
# ---------------- DETECCIÓN DE PATRONES DE VELA ----------------
def detectar_patron_velas(df):
    """
    Detecta patrones básicos de vela: martillo, shooting star, engulfing, doji.
    Retorna el nombre del patrón si hay coincidencia, o None.
    """
    if len(df) < 3:
        return None

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    o1, h1, l1, c1 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o2, h2, l2, c2 = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]

    cuerpo = abs(c1 - o1)
    rango = h1 - l1
    sombra_sup = h1 - max(c1, o1)
    sombra_inf = min(c1, o1) - l1

    # Evitar divisiones por cero
    if rango == 0:
        return None

    # 🕯️ Patrones simples
    if cuerpo < rango * 0.25 and sombra_inf > cuerpo * 2:
        return "HAMMER"
    elif cuerpo < rango * 0.25 and sombra_sup > cuerpo * 2:
        return "SHOOTING_STAR"
    elif c1 > o1 and o1 < c2 and c1 > o2:
        return "BULLISH_ENGULFING"
    elif c1 < o1 and o1 > c2 and c1 < o2:
        return "BEARISH_ENGULFING"
    elif abs(c1 - o1) <= rango * 0.1:
        return "DOJI"

    return None

# ---------------- BACKTEST SINCRONIZADO CON FLASK ----------------
def ejecutar_backtest(client, model, feature_cols, up_thr, down_thr):
    results = []
    total_symbols = len(SYMBOLS)

    for idx, sym in enumerate(SYMBOLS, 1):
        logging.info(f"\n⏳ [{idx}/{total_symbols}] Iniciando backtest para {sym}...")
        df = download_klines_safe(client, sym)
        df = build_features(df, sym, feature_cols)
        if df.empty:
            logging.warning(f"⚠️ {sym}: No hay datos suficientes.")
            continue

        capital = CAPITAL_INICIAL
        position = None
        entry_price = 0
        trades = wins = losses = 0
        gross_profit = gross_loss = 0
        equity_curve = [capital]

        for i in range(30, len(df)):  # pequeño warmup inicial para indicadores
            # 📊 IA: predicción de probabilidades
            X = df[feature_cols].iloc[[i]]
            probs = model.predict_proba(X)[0]
            prob_down, prob_neutral, prob_up = probs

            # 🧮 Indicadores técnicos adicionales
            price = df["Close"].iloc[i]
            atr = df["atr14"].iloc[i]
            adx = df["adx"].iloc[i]
            ema20 = df["ema_20"].iloc[i]
            ema50 = df["ema_50"].iloc[i]

            # 📌 Filtros mínimos de ATR y ADX
            atr_ratio = atr / price
            if atr_ratio < 0.0001 or adx < 10:
                ia_signal = "HOLD"
            else:
                # 🧠 Señal IA con umbrales
                if prob_up >= up_thr and (prob_up - prob_down) >= 0.02:
                    ia_signal = "BUY"
                elif prob_down >= down_thr and (prob_down - prob_up) >= 0.02:
                    ia_signal = "SELL"
                else:
                    ia_signal = "HOLD"

                # 📈 Confirmación técnica EMA
                if ia_signal == "BUY" and ema20 <= ema50:
                    ia_signal = "HOLD"
                elif ia_signal == "SELL" and ema20 >= ema50:
                    ia_signal = "HOLD"

                # 🕯️ Confirmación de patrones de vela
                patron = detectar_patron_velas(df.iloc[:i+1])
                if ia_signal == "BUY" and patron not in ["HAMMER", "BULLISH_ENGULFING"]:
                    ia_signal = "HOLD"
                elif ia_signal == "SELL" and patron not in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
                    ia_signal = "HOLD"

            # 🧭 TP / SL dinámicos basados en ATR
            sl_mult = 1.5 + (atr_ratio * 100)
            tp_mult = max(1.2, 2.2 - (atr_ratio * 50))
            sl = price - sl_mult * atr if ia_signal == "BUY" else price + sl_mult * atr
            tp = price + tp_mult * atr if ia_signal == "BUY" else price - tp_mult * atr

            # 📈 Ejecución de trade simulado
            if position is None and ia_signal in ["BUY", "SELL"]:
                position = "LONG" if ia_signal == "BUY" else "SHORT"
                entry_price = price
            elif position == "LONG":
                if price <= sl or price >= tp:
                    pnl = (price - entry_price)
                    capital += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        losses += 1
                        gross_loss += abs(pnl)
                    position = None
            elif position == "SHORT":
                if price >= sl or price <= tp:
                    pnl = (entry_price - price)
                    capital += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        losses += 1
                        gross_loss += abs(pnl)
                    position = None

            equity_curve.append(capital)

        # 📊 Métricas finales por símbolo
        accuracy = (wins / trades * 100) if trades > 0 else 0
        final_capital = capital
        drawdown = abs((np.array(equity_curve) - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)).max()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        sharpe_ratio = (
            (np.mean(np.diff(equity_curve)) / np.std(np.diff(equity_curve))) * np.sqrt(252)
            if len(equity_curve) > 1 and np.std(np.diff(equity_curve)) != 0
            else 0
        )

        logging.info(f"✔ {sym} — Trades: {trades} | Wins: {wins} | Losses: {losses} | PF: {profit_factor:.2f} | Capital final: {capital:.2f}")

        results.append({
            "symbol": sym,
            "capital_final": final_capital,
            "trades": trades,
            "accuracy": accuracy,
            "profit_factor": profit_factor,
            "drawdown": drawdown,
            "sharpe_ratio": sharpe_ratio,
            "UP": up_thr,
            "DOWN": down_thr
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        logging.info("\n📊 ================== RESUMEN FINAL ==================")
        logging.info(f"\n{df_results[['symbol','trades','accuracy','profit_factor','capital_final','sharpe_ratio']]}")
        logging.info("======================================================\n")
    return df_results

def ejecutar_backtest(client, model, feature_cols, up_thr, down_thr):
    ...
    # 📊 Métricas por símbolo
    df_results = pd.DataFrame(results)

    # 🏆 Ranking por profit factor
    df_sorted = df_results.sort_values(by="profit_factor", ascending=False)
    logging.info("🏁 MEJORES UMBRALES POR SÍMBOLO ==========================")
    for _, row in df_sorted.iterrows():
        sym = str(row.get("symbol", "N/A"))
        trades = int(float(row.get("trades", 0)))
        acc = round(float(row.get("accuracy", 0)), 2)
        pf = round(float(row.get("profit_factor", 0)), 2)
        gain = round(float(row.get("capital_final", CAPITAL_INICIAL)) - CAPITAL_INICIAL, 2)
        upv = float(row.get("UP", 0))
        downv = float(row.get("DOWN", 0))

        if trades == 0:
            logging.info(f"🚫 {sym} → Sin señales válidas")
        else:
            logging.info(
                f"{sym} → UP={upv:.2f} DOWN={downv:.2f} | Trades: {trades} | "
                f"Accuracy: {acc}% | PF={pf} | Ganancia: ${gain}"
            )
    logging.info("=========================================================\n")

    return df_results

# ---------------- OPTIMIZADOR DE UMBRALES ----------------
def optimizar_umbral(client, model, feature_cols):
    mejores = []
    for up_thr in np.arange(0.30, 0.61, 0.02):
        for down_thr in np.arange(0.30, 0.61, 0.02):
            df_results = ejecutar_backtest(client, model, feature_cols, up_thr, down_thr)
            if df_results.empty:
                continue
            avg_pf = df_results["profit_factor"].replace(np.inf, 10).mean()
            avg_sharpe = df_results["sharpe_ratio"].mean()
            score = avg_pf * 0.7 + avg_sharpe * 0.3
            mejores.append((up_thr, down_thr, avg_pf, avg_sharpe, score))

    if not mejores:
        logging.warning("⚠️ No se pudo optimizar umbrales.")
        return

    best = max(mejores, key=lambda x: x[4])
    logging.info(f"🏆 Mejor combinación: UP={best[0]:.2f} | DOWN={best[1]:.2f} | PF={best[2]:.2f} | Sharpe={best[3]:.2f}")
    df_best = ejecutar_backtest(client, model, feature_cols, best[0], best[1])
    df_best.to_csv("backtest_results_optim.csv", index=False)
    logging.info("✅ Backtest óptimo guardado en 'backtest_results_optim.csv'.")

# ---------------- JOB DIARIO ----------------
def job_diario():
    logging.info("⏰ Ejecutando backtest diario...")
    client = init_client(API_KEY, API_SECRET)
    model_dict = joblib.load(MODEL_FILE)
    model = model_dict["model"]
    feature_cols = model_dict["features"]
    optimizar_umbral(client, model, feature_cols)

# ---------------- MAIN ----------------


if __name__ == "__main__":
    logging.info("🚀 Sistema de backtest automático inicializado.")
    job_diario()  # ejecutar al iniciar
    schedule.every().day.at("00:00").do(job_diario)  # ejecutar cada día a medianoche

    while True:
        schedule.run_pending()
        time.sleep(60)

