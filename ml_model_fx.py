# ml_model_fx_optuna_multi_full_fixed.py
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

# ---------------- CONFIG ----------------
API_KEY = os.getenv("BINANCE_API_KEY") or "TU_API_KEY_TEMPORAL"
API_SECRET = os.getenv("BINANCE_API_SECRET") or "TU_API_SECRET_TEMPORAL"

SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT"]
INTERVAL = "1h"
LIMIT_PER_REQUEST = 1000
TOTAL_LIMIT_PER_SYMBOL = 5000
FUTURE_PERIOD = 3
RANDOM_STATE = 42
N_SPLITS = 5
SAVE_MODEL = "binance_ai_lgbm_optuna_multi_v2.pkl"

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
        "Open_time","Open","High","Low","Close","Volume",
        "Close_time","Quote_asset_volume","Number_of_trades",
        "Taker_buy_base","Taker_buy_quote","Ignore"
    ])
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms")
    df.set_index("Open_time", inplace=True)
    return df[["Open","High","Low","Close","Volume"]]

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
    df["vpt"] = (vol * (close.diff().fillna(0) / (close.shift(1)+1e-9))).cumsum()
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
    df_sym["target_up"] = (df_sym["future_ret"] > 0.001).astype(int)

    lag_feats = ["atr14","bb_pct","rsi14","stoch_k","stoch_d","macd","macd_signal","vpt","ama_cross","momentum","logret"]
    for f in lag_feats:
        for lag in (1,2,3):
            df_sym[f"{f}_lag{lag}"] = df_sym[f].shift(lag)
    df_sym.dropna(inplace=True)
    df_sym["symbol"] = sym
    frames.append(df_sym)
    time.sleep(0.1)

df_all = pd.concat(frames, axis=0).sort_index()
df_all = df_all[~df_all.index.duplicated(keep='first')]
print(f"\n✅ Dataset combinado: {df_all.shape[0]} filas, {df_all.shape[1]} columnas")

# ---------------- PREP X, y ----------------
drop_cols = ["future_close","future_ret","target_up"]
X = df_all[[c for c in df_all.columns if c not in drop_cols]]
y = df_all["target_up"]

# One-hot symbol
X = pd.get_dummies(X, columns=["symbol"], prefix="sym")

# Escalado
scaler = StandardScaler()
cols_to_scale = [c for c in ["vpt","momentum","logret"] if c in X.columns]
if cols_to_scale:
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

print(f"\n🔢 Features finales: {len(X.columns)} columnas")

# ---------------- OPTUNA ----------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
def objective(trial):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
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
        "class_weight": {0:1, 1:5},  # importante para clase minoritaria
        "seed": RANDOM_STATE
    }
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], eval_metric="auc", callbacks=[early_stopping(50)])
        preds = model.predict_proba(X_val)[:,1]
        aucs.append(roc_auc_score(y_val, preds))
    return float(np.mean(aucs))

print("\n🔎 Iniciando Optuna...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("✅ Mejor conjunto de hiperparámetros (Optuna):", best_params)

# ---------------- MODELO FINAL ----------------
final_params = best_params.copy()
final_params.update({
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 2000,
    "class_weight": {0:1, 1:5},
    "seed": RANDOM_STATE
})
final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, eval_metric="auc", callbacks=[log_evaluation(0)])
dump(final_model, SAVE_MODEL)
print(f"✅ Modelo guardado: {SAVE_MODEL}")

# ---------------- Feature importance ----------------
fi = pd.DataFrame({"feature": X.columns, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
fi.to_csv("feature_importance_optuna_multi.csv", index=False)
print("✅ Feature importance guardada: feature_importance_optuna_multi.csv")

# ---------------- PREDICCIONES ----------------
df_all["p_up"] = final_model.predict_proba(X)[:,1]
# ---------------- BÚSQUEDA ÓPTIMA DE UMBRALES POR SÍMBOLO ----------------
symbol_thresholds = {}
df_all["signal"] = 0  # inicializamos

for sym in SYMBOLS:
    df_sym = df_all[df_all["symbol"] == sym].copy()
    y_true = df_sym["target_up"].values
    y_proba = df_sym["p_up"].values

    best_acc = -1
    best_up, best_down = 0.8, 0.45
    for up in np.arange(0.6, 0.95, 0.01):
        for down in np.arange(0.05, 0.5, 0.01):
            signals = np.where(y_proba > up, 1, np.where(y_proba < down, -1, 0))
            idx = np.where(signals != 0)[0]
            if len(idx) == 0:
                continue
            acc = (signals[idx] == y_true[idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_up, best_down = up, down

    symbol_thresholds[sym] = {"UP": best_up, "DOWN": best_down}
    df_all.loc[df_all["symbol"] == sym, "signal"] = np.where(
        df_sym["p_up"] > best_up, 1,
        np.where(df_sym["p_up"] < best_down, -1, 0)
    )

print("\n🔎 Umbrales óptimos por símbolo:")
for sym, th in symbol_thresholds.items():
    print(f"{sym}: UP={th['UP']:.2f}, DOWN={th['DOWN']:.2f}")

# ---------------- BACKTEST MULTI-SÍMBOLO ----------------
df_bt = df_all.reset_index(drop=True)
df_bt["signal"] = np.where(df_bt["p_up"] > best_up, 1,
                           np.where(df_bt["p_up"] < best_down, -1, 0))
df_bt["next_ret"] = df_bt["Close"].pct_change().shift(-1).fillna(0)
df_bt["atr_pct"] = df_bt["atr14"] / df_bt["Close"]
tp_pct = TP_ATR_MULT * df_bt["atr_pct"]
sl_pct = SL_ATR_MULT * df_bt["atr_pct"]

long_mask = df_bt["signal"]==1
short_mask = df_bt["signal"]==-1
df_bt.loc[long_mask, "trade_ret"] = np.minimum(np.maximum(df_bt.loc[long_mask, "next_ret"], -sl_pct[long_mask]), tp_pct[long_mask])
df_bt.loc[short_mask, "trade_ret"] = np.minimum(np.maximum(-df_bt.loc[short_mask, "next_ret"], -sl_pct[short_mask]), tp_pct[short_mask])

df_bt["prev_signal"] = df_bt["signal"].shift(1).fillna(0)
df_bt["trade_entry"] = (df_bt["signal"] != df_bt["prev_signal"]) & (df_bt["signal"] != 0)
df_bt["trade_cost"] = df_bt["trade_entry"].astype(int) * (COMMISSION + SLIPPAGE)
df_bt["strategy_ret"] = df_bt["trade_ret"].fillna(0) - df_bt["trade_cost"]
df_bt["cum_strategy"] = (1 + df_bt["strategy_ret"]).cumprod()
df_bt["cum_market"] = (1 + df_bt["next_ret"]).cumprod()

total_return = df_bt["cum_strategy"].iloc[-1]
market_return = df_bt["cum_market"].iloc[-1]
periods_per_year = 252*24
ann_ret = df_bt["strategy_ret"].mean() * periods_per_year
ann_vol = df_bt["strategy_ret"].std() * np.sqrt(periods_per_year)
sharpe = ann_ret / (ann_vol + 1e-9)
running_max = df_bt["cum_strategy"].cummax()
max_dd = ((running_max - df_bt["cum_strategy"]) / (running_max + 1e-9)).max()

print("\n=== BACKTEST MULTI-SÍMBOLO ===")
print(f"Rentabilidad estrategia final: {total_return:.4f}")
print(f"Rentabilidad Buy & Hold: {market_return:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
print(f"Sharpe estimado: {sharpe:.2f}")

plt.figure(figsize=(12,6))
plt.plot(df_bt["cum_strategy"], label="Estrategia (neto)")
plt.plot(df_bt["cum_market"], label="Buy & Hold")
plt.legend()
plt.title("Equity Curve Multi-Símbolo (aprox.)")
plt.show()

# ---------------- EVALUACIÓN ----------------
df_eval = df_all.loc[~df_all.index.duplicated(keep='first')].copy()
df_eval["signal"] = np.where(df_eval["p_up"] > best_up, 1, np.where(df_eval["p_up"] < best_down, -1, 0))
y_true = df_eval["target_up"]
y_pred = (df_eval["signal"]==1).astype(int)

print("\n--- EVALUACIÓN GLOBAL MULTI-SÍMBOLO ---")
print(classification_report(y_true, y_pred, digits=4))
auc_global = roc_auc_score(y_true, df_eval["p_up"])
print("ROC AUC global:", round(auc_global,4))

print("\n--- EVALUACIÓN POR SÍMBOLO ---")
for s in SYMBOLS:
    df_s = df_eval[df_eval["symbol"]==s]
    y_true_s = df_s["target_up"]
    y_pred_s = (df_s["signal"]==1).astype(int)
    print(f"\n{s}:")
    print(classification_report(y_true_s, y_pred_s, digits=4))
    auc_s = roc_auc_score(y_true_s, df_s["p_up"])
    print("ROC AUC:", round(auc_s,4))

# ---------------- GUARDADO ----------------
df_bt.to_csv("binance_predictions_optuna_multi_v2.csv", index=False)
fi.to_csv("feature_importance_optuna_multi_v2.csv", index=False)
study.trials_dataframe().to_csv("optuna_trials_multi_v2.csv", index=False)
dump({"model": final_model, "scaler": scaler, "features": X.columns.tolist()}, SAVE_MODEL)
print("✅ Archivos guardados: modelo, predictions, feature importance, optuna trials")
