import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 📊 Configuración
TRADE_LOG_FILE = Path("trade_log.csv")
OUTPUT_FILE = Path("best_thresholds_auto.json")

# Parámetros mínimos para filtrar pares
MIN_TRADES = 5
MIN_WINRATE = 0.5     # 50%
MIN_PROFIT_FACTOR = 1.2

def load_trade_log():
    if not TRADE_LOG_FILE.exists():
        logging.warning("⚠️ No existe trade_log.csv")
        return pd.DataFrame()

    df = pd.read_csv(TRADE_LOG_FILE)
    # Asegurar que las columnas clave existan
    required_cols = {"symbol", "signal", "entry_price", "exit_price", "pnl_usd", "pnl_pct", "prob_up", "prob_down", "type_signal"}
    missing = required_cols - set(df.columns)
    if missing:
        logging.error(f"❌ Faltan columnas en el log: {missing}")
        return pd.DataFrame()

    return df

def compute_metrics(df):
    metrics = []
    for symbol, data in df.groupby("symbol"):
        total_trades = len(data)
        wins = (data["pnl_usd"] > 0).sum()
        losses = (data["pnl_usd"] <= 0).sum()

        total_profit = data[data["pnl_usd"] > 0]["pnl_usd"].sum()
        total_loss = abs(data[data["pnl_usd"] <= 0]["pnl_usd"].sum())
        profit_factor = total_profit / total_loss if total_loss != 0 else float("inf")
        winrate = wins / total_trades if total_trades > 0 else 0

        avg_rr = (data["pnl_pct"]).mean() / 100
        max_drawdown = data["pnl_usd"].cumsum().min()

        metrics.append({
            "symbol": symbol,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_rr": round(avg_rr, 3),
            "max_drawdown": round(max_drawdown, 2),
        })

    return pd.DataFrame(metrics)

def select_best_symbols(metrics_df):
    if metrics_df.empty:
        return []

    filtered = metrics_df[
        (metrics_df["total_trades"] >= MIN_TRADES) &
        (metrics_df["winrate"] >= MIN_WINRATE) &
        (metrics_df["profit_factor"] >= MIN_PROFIT_FACTOR)
    ]
    return filtered["symbol"].tolist()

def update_best_thresholds(best_symbols):
    """
    🔁 Actualiza el archivo de umbrales automáticos en función de los pares con mejor rendimiento
    """
    thresholds = {sym: {"UP": 0.5, "DOWN": 0.56, "DIFF": 0.05} for sym in best_symbols}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)

    logging.info(f"✅ Thresholds auto actualizados: {best_symbols}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = load_trade_log()
    if df.empty:
        logging.warning("⚠️ No hay datos para analizar.")
        return

    metrics = compute_metrics(df)
    logging.info("\n📊 Métricas de desempeño por símbolo:\n" + str(metrics))

    best_symbols = select_best_symbols(metrics)
    if not best_symbols:
        logging.warning("⚠️ No se encontraron símbolos con buen rendimiento.")
    else:
        logging.info(f"🏆 Mejores símbolos detectados: {best_symbols}")
        update_best_thresholds(best_symbols)

if __name__ == "__main__":
    main()
