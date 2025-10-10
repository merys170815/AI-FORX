import pandas as pd
import numpy as np
import yfinance as yf
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# =========================
# 1. DESCARGAR O CARGAR DATOS
# =========================
try:
    print("Descargando datos históricos de EURUSD...")
    df = yf.download("EURUSD=X", period="2y", interval="1h", progress=False).reset_index()
    df.to_csv("EURUSD_H1.csv", index=False)
    print("Archivo 'EURUSD_H1.csv' creado con éxito.")
except Exception as e:
    print("Error descargando datos, cargando CSV local...")
    print(e)
    df = pd.read_csv("EURUSD_H1.csv")

print("Datos cargados, preparando indicadores...")

# =========================
# 2. INDICADORES
# =========================
df.columns = [c.lower() for c in df.columns]
df = df.dropna()
df["ema10"] = ta.trend.EMAIndicator(df["close"], 10).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
df = df.dropna()
print("Indicadores calculados.")

# =========================
# 3. OBJETIVO
# =========================
df["target"] = np.where((df["close"].shift(-6) - df["close"]) / df["close"] > 0.001, 1, 0)
df = df.dropna()
print("Variable objetivo creada.")

# =========================
# 4. ENTRENAMIENTO
# =========================
features = ["ema10", "ema50", "rsi"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)
print("Modelo entrenado.")

# =========================
# 5. EVALUACIÓN
# =========================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

print("\n=== RESULTADOS DEL MODELO ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# =========================
# 6. BACKTEST
# =========================
df_test = X_test.copy()
df_test["close"] = df["close"].iloc[X_train.shape[0]:].values
df_test["signal"] = y_pred
df_test["return"] = df_test["close"].pct_change().shift(-1)
df_test["strategy_return"] = df_test["return"] * df_test["signal"]
df_test["cum_strategy"] = (1 + df_test["strategy_return"]).cumprod()
df_test["cum_market"] = (1 + df_test["return"]).cumprod()

print("\n=== BACKTEST ===")
print(f"Rentabilidad estrategia final: {df_test['cum_strategy'].iloc[-1]:.4f}")
print(f"Rentabilidad Buy & Hold final: {df_test['cum_market'].iloc[-1]:.4f}")

# =========================
# 7. GRAFICO
# =========================
plt.figure(figsize=(12,6))
plt.plot(df_test.index, df_test["cum_strategy"], label="Estrategia ML")
plt.plot(df_test.index, df_test["cum_market"], label="Buy & Hold")
plt.title("Backtesting Estrategia ML vs Buy & Hold")
plt.xlabel("Velas")
plt.ylabel("Capital acumulado")
plt.legend()
plt.show()



if __name__ == "__main__":
    main()
















