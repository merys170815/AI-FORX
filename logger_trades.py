import csv
import os
from datetime import datetime

# 📁 Nombre del archivo de registro
TRADE_LOG_FILE = "trade_log.csv"

# 🧾 Encabezados del archivo
HEADERS = [
    "timestamp",
    "symbol",
    "signal",
    "entry_price",
    "exit_price",
    "pnl_usd",
    "pnl_pct",
    "prob_up",
    "prob_down",
    "type_signal",
    "narrative"
]

# 📌 Inicializa el archivo si no existe
if not os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)


def log_trade(symbol, signal, entry_price, exit_price, pnl_usd, pnl_pct,
              prob_up, prob_down, type_signal, narrative):
    """
    📊 Registra un trade en trade_log.csv con codificación UTF-8
    """
    timestamp = datetime.now().isoformat(timespec='seconds')
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            symbol,
            signal,
            round(entry_price, 4),
            round(exit_price, 4),
            round(pnl_usd, 4),
            round(pnl_pct, 2),
            round(prob_up, 2),
            round(prob_down, 2),
            type_signal,
            narrative
        ])
    print(f"✅ Trade registrado en {TRADE_LOG_FILE}: {symbol} {signal} (PnL: {pnl_usd:.2f} USD)")
