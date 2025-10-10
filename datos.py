import yfinance as yf
import pandas as pd
import numpy as np
import ta  # librería de indicadores técnicos

# === DESCARGAR DATOS HISTÓRICOS MÁXIMOS ===
print("Descargando datos históricos máximos de EURUSD...")
data = yf.download("EURUSD=X", period="max", interval="1h", auto_adjust=True)
df = data.reset_index()

# Renombrar columnas para mayor claridad
df.columns = ["date", "open", "high", "low", "close", "volume"]

# === INDICADORES TÉCNICOS ===
# Tendencia
df['SMA14'] = ta.trend.sma_indicator(df['close'], window=14)
df['EMA14'] = ta.trend.ema_indicator(df['close'], window=14)
df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
df['MACD'] = ta.trend.macd(df['close'])
df['MACD_signal'] = ta.trend.macd_signal(df['close'])

# Momentum
df['RSI14'] = ta.momentum.rsi(df['close'], window=14)
df['Momentum'] = ta.momentum.roc(df['close'], window=14)

# Volatilidad
bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['BB_high'] = bb.bollinger_hband()
df['BB_low'] = bb.bollinger_lband()
df['ATR14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

# === FEATURES DE LAGS (precios y volumen) ===
lags = [1, 2, 3, 6, 12]  # horas anteriores
for lag in lags:
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

# === VARIABLE OBJETIVO ===
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# === LIMPIAR NA ===
df = df.dropna()

# === GUARDAR DATASET FINAL EN CSV ===
df.to_csv("EURUSD_features_full.csv", index=False)
print("Archivo 'EURUSD_features_full.csv' creado con éxito.")
print("Primeras filas del dataset final:")
print(df.head())


