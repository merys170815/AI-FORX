# ml_model_fx_ai_pro.py
# Trading FX inteligente avanzado con RandomForest y tamaño de posición dinámico
# Requisitos: pandas, numpy, yfinance, ta, scikit-learn, joblib, matplotlib, seaborn
# Ejecutar: python ml_model_fx_ai_pro.py

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------- CONFIG ----------
TICKER = "EURUSD=X"
PERIOD = "12mo"
INTERVAL = "1h"
FUTURE_PERIOD = 1       # barras hacia adelante
P_UP_THRESHOLD = 0.65   # LONG
P_DOWN_THRESHOLD = 0.35 # SHORT
SAVE_MODEL = "fx_model_ai_pro.pkl"

# ---------- 1) Descargar datos ----------
print("Descargando datos...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).capitalize() for c in df.columns]
df.dropna(inplace=True)
if df.empty:
    raise SystemExit("No se obtuvieron datos")

# ---------- 2) Crear features ----------
# EMAs
df["ema20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
df["ema200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
# RSI
df["rsi14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
# MACD
macd = ta.trend.MACD(df["Close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"] = df["macd"] - df["macd_signal"]
# ATR
df["atr14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
# Momentum / returns
df["ret_1"] = df["Close"].pct_change(1)
df["ret_3"] = df["Close"].pct_change(3)
df["vol_rolling"] = df["ret_1"].rolling(24).std()
# Pivots
df["sr_high20"] = df["High"].rolling(20).max()
df["sr_low20"] = df["Low"].rolling(20).min()
# Lags
for lag in (1,2,3):
    df[f"close_lag_{lag}"] = df["Close"].shift(lag)
    df[f"rsi_lag_{lag}"] = df["rsi14"].shift(lag)

# ---------- 3) Target ----------
df["future_close"] = df["Close"].shift(-FUTURE_PERIOD)
df["target_up"] = (df["future_close"] > df["Close"]).astype(int)
df.dropna(inplace=True)
print(f"Filas disponibles: {len(df)}")

# ---------- 4) Features y split ----------
feature_cols = [
    "ema20","ema50","ema200","rsi14","macd","macd_hist","atr14",
    "ret_1","ret_3","vol_rolling","sr_high20","sr_low20",
    "close_lag_1","close_lag_2","close_lag_3","rsi_lag_1","rsi_lag_2","rsi_lag_3"
]
X = df[feature_cols]
y = df["target_up"]

split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------- 5) Entrenamiento ----------
print("Entrenando RandomForest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

# ---------- 6) Evaluación ----------
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- EVALUATION (test) ---")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (test)")
plt.show()

# Feature importances
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
fi.head(15).plot(kind="barh", figsize=(8,6))
plt.title("Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# ---------- 7) Backtest inteligente con tamaño dinámico ----------
df_test = X_test.copy()
df_test["close"] = df["Close"].iloc[split_idx:]
df_test["p_up"] = y_proba
df_test["signal"] = 0

# Señales LONG/SHORT + tamaño de posición dinámico
df_test["signal"] = np.where(
    (df_test["p_up"] > P_UP_THRESHOLD),
    df_test["p_up"],   # LONG ponderado por confianza
    np.where(df_test["p_up"] < P_DOWN_THRESHOLD, df_test["p_up"]-1, 0)  # SHORT ponderado
)

# Retornos ajustados por tamaño de posición
df_test["strategy_return"] = df_test["close"].pct_change().shift(-1) * df_test["signal"]
df_test["cum_strategy"] = (1 + df_test["strategy_return"].fillna(0)).cumprod()
df_test["cum_market"] = (1 + df_test["close"].pct_change().shift(-1).fillna(0)).cumprod()

# Métricas de riesgo
cum = df_test["cum_strategy"]
drawdown = (cum.cummax() - cum).max()
sharpe = df_test["strategy_return"].mean() / (df_test["strategy_return"].std() + 1e-9) * np.sqrt(252*24)

print("\n=== BACKTEST RESULTADOS ===")
print(f"Rentabilidad estrategia final: {df_test['cum_strategy'].iloc[-1]:.4f}")
print(f"Rentabilidad Buy & Hold final: {df_test['cum_market'].iloc[-1]:.4f}")
print(f"Drawdown máximo: {drawdown:.4f}")
print(f"Sharpe ratio estimado: {sharpe:.2f}")

# Equity curve
plt.figure(figsize=(12,6))
plt.plot(df_test["cum_strategy"], label="Estrategia AI Pro", color="blue")
plt.plot(df_test["cum_market"], label="Buy & Hold", color="orange")
plt.title("Equity Curve")
plt.xlabel("Barra")
plt.ylabel("Capital acumulado")
plt.legend()
plt.show()

# ---------- 8) Guardar modelo y predicciones ----------
dump(rf, SAVE_MODEL)
preds_df = X_test.copy()
preds_df["y_true"] = y_test
preds_df["y_pred"] = y_pred
preds_df["y_proba"] = y_proba
preds_df.to_csv("fx_predictions_ai_pro.csv", index=False)

print(f"✅ Modelo guardado: {SAVE_MODEL}")
print("✅ Predicciones guardadas en fx_predictions_ai_pro.csv")

print("\nUSO EN PRODUCCIÓN:")
print(f"- Carga '{SAVE_MODEL}' con joblib.load")
print("- Calcula features idénticos y llama model.predict_proba(X)[0,1]")
print(f"- LONG si p_up > {P_UP_THRESHOLD} y tendencia positiva")
print(f"- SHORT si p_up < {P_DOWN_THRESHOLD} y tendencia negativa")
print("- Tamaño de posición proporcional a la confianza y volatilidad")
