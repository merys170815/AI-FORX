import os
import joblib
import pandas as pd
import numpy as np
import ta
from binance.client import Client
from datetime import datetime
import logging
from itertools import product

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


def download_klines_safe(client, sym, interval, limit):
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


def ichimoku(df):
    high, low, close = df['High'], df['Low'], df['Close']
    ich = ta.trend.IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
    return ich.ichimoku_conversion_line(), ich.ichimoku_base_line(), ich.ichimoku_a(), ich.ichimoku_b(), close.shift(-26)


def compute_rsi(close, period=14):
    return ta.momentum.RSIIndicator(close, window=period).rsi()


def compute_atr(df, window=14):
    return ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()


def compute_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return abs(drawdown.min())


# ---------------- DIAGNÓSTICO ----------------
def diagnostico_modelo(client, model, feature_cols, symbols):
    logging.info("\n🔍 Diagnóstico del modelo IA...")
    for sym in symbols:
        df = download_klines_safe(client, sym, INTERVAL, 300)
        if df.empty:
            continue

        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df)
        df["tenkan"] = tenkan
        df["kijun"] = kijun
        df["senkou_a"] = senkou_a
        df["senkou_b"] = senkou_b
        df["chikou"] = chikou
        df["rsi"] = compute_rsi(df['Close'])
        df["atr"] = compute_atr(df)
        df.dropna(inplace=True)

        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        df.ffill().bfill(inplace=True)

        X = df[feature_cols].iloc[[-1]]
        probs = model.predict_proba(X)[0]
        prob_down, prob_neutral, prob_up = probs
        logging.info(f"📈 {sym}: ↓ {prob_down:.2f} | → {prob_neutral:.2f} | ↑ {prob_up:.2f}")


# ---------------- BACKTEST ----------------
def ejecutar_backtest(client, model, feature_cols, up_thr, down_thr):
    results = []

    for sym in SYMBOLS:
        logging.info(f"\n📊 Backtesting {sym} (UP={up_thr}, DOWN={down_thr})...")

        df = download_klines_safe(client, sym, INTERVAL, HISTORICAL_LIMIT)
        if df.empty:
            continue

        df_feat = df.copy()
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(df_feat)
        df_feat["tenkan"] = tenkan
        df_feat["kijun"] = kijun
        df_feat["senkou_a"] = senkou_a
        df_feat["senkou_b"] = senkou_b
        df_feat["chikou"] = chikou
        df_feat["rsi"] = compute_rsi(df_feat['Close'])
        df_feat["atr"] = compute_atr(df_feat)
        df_feat.dropna(inplace=True)

        for c in feature_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        df_feat.ffill().bfill(inplace=True)

        capital = CAPITAL_INICIAL
        position = None
        entry_price = 0
        trades = wins = losses = 0
        gross_profit = gross_loss = 0
        equity_curve = [capital]

        for i in range(1, len(df_feat)):
            X = df_feat[feature_cols].iloc[[i]]
            probs = model.predict_proba(X)[0]
            prob_down, prob_neutral, prob_up = probs

            if prob_up >= up_thr:
                ia_signal = "BUY"
            elif prob_down >= down_thr:
                ia_signal = "SELL"
            else:
                last_close = df_feat["Close"].iloc[i]
                prev_close = df_feat["Close"].iloc[i - 1]
                ia_signal = "BUY" if last_close > prev_close else ("SELL" if last_close < prev_close else "HOLD")

            price = df_feat["Close"].iloc[i]
            atr = df_feat["atr"].iloc[i]

            # Dinámica de SL/TP según volatilidad reciente
            atr_ratio = atr / price
            sl_mult = 1.5 + (atr_ratio * 100)
            tp_mult = 2.2 - (atr_ratio * 50)
            tp_mult = max(1.2, tp_mult)

            if ia_signal == "BUY":
                sl = price - sl_mult * atr
                tp = price + tp_mult * atr
            elif ia_signal == "SELL":
                sl = price + sl_mult * atr
                tp = price - tp_mult * atr
            else:
                continue

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

        accuracy = (wins / trades * 100) if trades > 0 else 0
        final_capital = capital
        drawdown = compute_drawdown(equity_curve)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        sharpe_ratio = (
            (np.mean(np.diff(equity_curve)) / np.std(np.diff(equity_curve))) * np.sqrt(252)
            if len(equity_curve) > 1 and np.std(np.diff(equity_curve)) != 0
            else 0
        )

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

    return pd.DataFrame(results)


# ---------------- OPTIMIZADOR ----------------
def optimizar_umbral(client, model, feature_cols):
    mejores = []
    for up_thr, down_thr in product(np.arange(0.45, 0.56, 0.02), np.arange(0.45, 0.56, 0.02)):
        df_results = ejecutar_backtest(client, model, feature_cols, up_thr, down_thr)
        avg_pf = df_results["profit_factor"].replace(np.inf, 10).mean()
        avg_sharpe = df_results["sharpe_ratio"].mean()
        score = avg_pf * 0.7 + avg_sharpe * 0.3
        mejores.append((up_thr, down_thr, avg_pf, avg_sharpe, score))
        logging.info(f"Evaluado (UP={up_thr:.2f}, DOWN={down_thr:.2f}) → PF={avg_pf:.2f}, Sharpe={avg_sharpe:.2f}")

    best = max(mejores, key=lambda x: x[4])
    logging.info(f"\n🏆 Mejor combinación: UP={best[0]:.2f} | DOWN={best[1]:.2f} | PF={best[2]:.2f} | Sharpe={best[3]:.2f}")

    df_best = ejecutar_backtest(client, model, feature_cols, best[0], best[1])
    df_best.to_csv("backtest_results_optim.csv", index=False)
    logging.info("✅ Backtest óptimo guardado en 'backtest_results_optim.csv'.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    client = init_client(API_KEY, API_SECRET)

    logging.info("Cargando modelo IA...")
    model_dict = joblib.load(MODEL_FILE)
    model = model_dict["model"]
    feature_cols = model_dict["features"]
    logging.info(f"✅ Modelo cargado con {len(feature_cols)} features y clases: {model.classes_}")

    diagnostico_modelo(client, model, feature_cols, SYMBOLS)

    logging.info("\n🚀 Iniciando optimización automática de umbrales...")
    optimizar_umbral(client, model, feature_cols)
