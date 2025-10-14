# backtest_analysis_complete_realistic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INITIAL_CAPITAL = 10000
TRADES_LOG = "trades_log.csv"

# ---------------- CARGAR TRADES ----------------
if not os.path.exists(TRADES_LOG):
    raise FileNotFoundError(f"No se encontró el archivo {TRADES_LOG}")

# Leer CSV, saltando líneas corruptas
df_all = pd.read_csv(TRADES_LOG, on_bad_lines='skip')

# Asegurar columnas correctas
if df_all.shape[1] == 7:
    df_all.columns = ["timestamp", "symbol", "signal", "prob", "price", "stop_loss", "take_profit"]
    df_all["extra"] = ""  # columna extra vacía si falta
elif df_all.shape[1] == 8:
    df_all.columns = ["timestamp", "symbol", "signal", "prob", "price", "stop_loss", "take_profit", "extra"]
else:
    raise ValueError(f"El archivo tiene un número inesperado de columnas: {df_all.shape[1]}")


# ---------------- CALCULO DE PnL REALISTA ----------------
def calc_pnl(row):
    price = row["price"]
    sl = row["stop_loss"]
    tp = row["take_profit"]
    signal = row["signal"]

    # Si no hay TP o SL, asumimos PnL = 0
    if pd.isna(tp) or pd.isna(sl):
        return 0.0

    if signal == "BUY":
        return tp - price
    elif signal == "SELL":
        return price - tp
    else:
        return 0.0


df_all["pnl"] = df_all.apply(calc_pnl, axis=1)

# ---------------- ANALISIS POR SIMBOLO ----------------
summary = []
for sym in SYMBOLS:
    df = df_all[df_all["symbol"] == sym].copy()
    if df.empty:
        continue

    total_trades = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0

    # Capital acumulado y drawdown
    capital_curve = INITIAL_CAPITAL + df["pnl"].cumsum()
    max_dd = INITIAL_CAPITAL - capital_curve.min() if not capital_curve.empty else 0

    # Sharpe ratio simple
    returns = df["pnl"] / INITIAL_CAPITAL
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    summary.append({
        "symbol": sym,
        "capital_final": capital_curve.iloc[-1] if not capital_curve.empty else INITIAL_CAPITAL,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe
    })

    # Graficar equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(capital_curve, label=f"Equity {sym}")
    plt.title(f"Equity Curve {sym}")
    plt.xlabel("Trade #")
    plt.ylabel("Capital USD")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"equity_curve_{sym}.png")
    plt.close()

# ---------------- RESUMEN GLOBAL ----------------
df_summary = pd.DataFrame(summary)
df_summary.to_csv("backtest_summary_realistic.csv", index=False)
print("✅ Análisis completo finalizado. Resumen guardado en backtest_summary_realistic.csv")
print(df_summary)
