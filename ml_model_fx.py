# ml_model_fx_optuna_multi_full_fixed_multiclass.py
import os, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from joblib import dump
from datetime import datetime
from binance.client import Client
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")
import os

# ---------------- CONFIG ----------------
API_KEY = os.getenv("BINANCE_API_KEY") or "TU_API_KEY_TEMPORAL"
API_SECRET = os.getenv("BINANCE_API_SECRET") or "TU_API_SECRET_TEMPORAL"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
LIMIT_PER_REQUEST = 1000
TOTAL_LIMIT_PER_SYMBOL = 5000
FUTURE_PERIOD = 3
RANDOM_STATE = 42
N_SPLITS = 5
SAVE_MODEL = "binance_ai_lgbm_optuna_multi_v2_multiclass.pkl"

COMMISSION = 0.0004
SLIPPAGE = 0.0005
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0

# ---------------- CONNECT ----------------
try:
    client = Client(API_KEY, API_SECRET)
    client.futures_ping()
    print("✅ Conexión a Binance Futures OK")
except Exception as e:
    print("⚠️ No se pudo conectar a Binance:", e)
    raise RuntimeError("Conexión Binance requerida")


# ---------------- DOWNLOAD ----------------
def download_klines_safe(symbol, interval, total_limit=1500, limit_per_request=1000):
    all_klines = []
    remaining = total_limit
    last_ts = None
    attempts = 0
    while remaining > 0 and attempts < 50:
        batch = min(remaining, limit_per_request)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if last_ts:
            params["endTime"] = last_ts - 1
        kl = client.futures_klines(**params)
        if not kl:
            break
        all_klines = kl + all_klines
        remaining -= len(kl)
        last_ts = int(kl[0][0])
        attempts += 1
        time.sleep(0.2)
    df = pd.DataFrame(all_klines, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Number_of_trades",
        "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    df.set_index("Open_time", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ---------------- FEATURES ----------------
def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["ama_cross"] = df["ema_10"] - df["ema_20"]
    df["momentum"] = close.diff().fillna(0)
    df["atr14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-9)
    df["rsi14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["vpt"] = (vol * (close.diff().fillna(0) / (close.shift(1) + 1e-9))).cumsum()
    df["logret"] = np.log(close).diff().fillna(0)
    return df


# ---------------- DESCARGA MULTI-SÍMBOLO ----------------
frames = []
print("\n📥 Descargando y preparando datos por símbolo...")
for sym in SYMBOLS:
    print(f"  ▪️ {sym}: descargando {TOTAL_LIMIT_PER_SYMBOL} velas...")
    df_sym = download_klines_safe(sym, INTERVAL, TOTAL_LIMIT_PER_SYMBOL, LIMIT_PER_REQUEST)
    df_sym = add_indicators(df_sym)
    df_sym["future_close"] = df_sym["Close"].shift(-FUTURE_PERIOD)
    df_sym["future_ret"] = (df_sym["future_close"] - df_sym["Close"]) / df_sym["Close"]

    # Target multiclass
    df_sym["target"] = 0
    df_sym.loc[df_sym["future_ret"] > 0.001, "target"] = 1
    df_sym.loc[df_sym["future_ret"] < -0.001, "target"] = -1

    lag_feats = ["atr14", "bb_pct", "rsi14", "stoch_k", "stoch_d", "macd", "macd_signal", "vpt", "ama_cross",
                 "momentum", "logret"]
    for f in lag_feats:
        for lag in (1, 2, 3):
            df_sym[f"{f}_lag{lag}"] = df_sym[f].shift(lag)
    df_sym.dropna(inplace=True)
    df_sym["symbol"] = sym
    frames.append(df_sym)
    time.sleep(0.1)

df_all = pd.concat(frames, axis=0).sort_index()
df_all = df_all[~df_all.index.duplicated(keep='first')]
print(f"\n✅ Dataset combinado: {df_all.shape[0]} filas, {df_all.shape[1]} columnas")

# ---------------- PREP X, y ----------------
drop_cols = ["future_close", "future_ret", "target"]
X = df_all[[c for c in df_all.columns if c not in drop_cols]]
y = df_all["target"]

# One-hot symbol
X = pd.get_dummies(X, columns=["symbol"], prefix="sym")

# Escalado
scaler = StandardScaler()
cols_to_scale = [c for c in ["vpt", "momentum", "logret"] if c in X.columns]
if cols_to_scale:
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

print(f"\n🔢 Features finales: {len(X.columns)} columnas")

# ---------------- OPTUNA ----------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "seed": RANDOM_STATE
    }
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="multi_logloss", callbacks=[early_stopping(50)])
        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    return float(np.mean(scores))


print("\n🔎 Iniciando Optuna...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("✅ Mejor conjunto de hiperparámetros (Optuna):", best_params)

# ---------------- MODELO FINAL ----------------
final_params = best_params.copy()
final_params.update({
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 2000,
    "seed": RANDOM_STATE
})
final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, eval_metric="multi_logloss", callbacks=[log_evaluation(0)])
dump(final_model, SAVE_MODEL)
print(f"✅ Modelo guardado: {SAVE_MODEL}")

# ---------------- Feature importance ----------------
fi = pd.DataFrame({"feature": X.columns, "importance": final_model.feature_importances_}).sort_values("importance",
                                                                                                      ascending=False)
fi.to_csv("feature_importance_optuna_multi_multiclass.csv", index=False)
print("✅ Feature importance guardada: feature_importance_optuna_multi_multiclass.csv")

# ---------------- PREDICCIONES ----------------
df_all["pred_class"] = final_model.predict(X)
df_all["p_up"] = final_model.predict_proba(X)[:, 2]  # probabilidad de clase 1

# ---------------- EVALUACIÓN GLOBAL MULTICLAS ----------------
y_true = df_all["target"]
y_pred = df_all["pred_class"]

print("\n--- EVALUACIÓN GLOBAL MULTI-SÍMBOLO (3 CLASES) ---")
print(classification_report(y_true, y_pred, digits=4))

# ---------------- EVALUACIÓN POR SÍMBOLO ----------------
print("\n--- EVALUACIÓN POR SÍMBOLO (3 CLASES) ---")
for s in SYMBOLS:
    df_s = df_all[df_all["symbol"] == s]
    y_true_s = df_s["target"]
    y_pred_s = df_s["pred_class"]
    print(f"\n{s}:")
    print(classification_report(y_true_s, y_pred_s, digits=4))

# ---------------- GUARDADO ----------------
df_all.to_csv("binance_predictions_optuna_multi_multiclass.csv", index=False)
fi.to_csv("feature_importance_optuna_multi_multiclass.csv", index=False)
study.trials_dataframe().to_csv("optuna_trials_multi_multiclass.csv", index=False)
dump({"model": final_model, "scaler": scaler, "features": X.columns.tolist()}, SAVE_MODEL)
print("✅ Archivos guardados: modelo, predictions, feature importance, optuna trials")

# ===============================
# 🚀 Auto reentrenamiento programado cada 4 horas
# ===============================
import threading

def auto_retrain(interval_hours=4):
    while True:
        print(f"\n🕒 Esperando {interval_hours} horas para reentrenar nuevamente...")
        time.sleep(interval_hours * 3600)  # 4 horas por defecto
        try:
            print("\n⚡ Reentrenamiento automático iniciado...")
            os.system(f"python {__file__}")  # vuelve a ejecutar este mismo script
            print("✅ Reentrenamiento completado.")
        except Exception as e:
            print(f"❌ Error en reentrenamiento automático: {e}")

# Lanzar el hilo automático
threading.Thread(target=auto_retrain, daemon=True).start()
