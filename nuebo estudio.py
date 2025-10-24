import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from binance.client import Client
from math import isnan
import warnings

# 🧠 Importa tu lógica real de producción
from app import compute_signal_for_symbol as compute_signal_original, API_KEY, API_SECRET

# ---------------- CONFIG ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
INTERVAL = "1h"
CAPITAL_INICIAL = 10000
RISK_PER_TRADE = 0.01
LOOKAHEAD = 20  # velas después de la entrada para evaluar TP/SL
HISTORICAL_DAYS = 30

warnings.filterwarnings("ignore", category=FutureWarning)
client = Client(API_KEY, API_SECRET)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------- WRAPPER PARA BACKTEST ----------------
def compute_signal_for_symbol_backtest(symbol, balance, df_feat):
    """Parchea la función real para usar df histórico en lugar de velas en vivo."""
    import app
    original_download = app.download_klines_safe

    def fake_download(sym):
        return df_feat

    app.download_klines_safe = fake_download
    result = compute_signal_original(symbol, balance=balance)
    app.download_klines_safe = original_download
    return result


# ---------------- FUNCIONES BASE ----------------
def download_klines(symbol):
    """Descarga velas históricas de Binance Futures para ~1 año."""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=HISTORICAL_DAYS)).timestamp() * 1000)

    klines = []
    temp_start = start_time

    while temp_start < end_time:
        batch = client.futures_klines(
            symbol=symbol, interval=INTERVAL, startTime=temp_start, endTime=end_time, limit=1500
        )
        if not batch:
            break
        klines.extend(batch)
        temp_start = batch[-1][6]  # close_time siguiente

    df = pd.DataFrame(klines, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
        "Quote_asset_volume", "Number_of_trades", "Taker_buy_base", "Taker_buy_quote", "Ignore"
    ])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    return df


def simular_trade(entry, sl, tp, future_df, size, signal):
    for _, row in future_df.iterrows():
        high = row["High"]
        low = row["Low"]

        if signal == "COMPRAR":
            if high >= tp:
                return {"result": "WIN", "profit": (tp - entry) * size}
            if low <= sl:
                return {"result": "LOSS", "profit": (sl - entry) * size}
        else:  # VENTA
            if low <= tp:
                return {"result": "WIN", "profit": (entry - tp) * size}
            if high >= sl:
                return {"result": "LOSS", "profit": (entry - sl) * size}

    last_price = future_df.iloc[-1]["Close"]
    profit = (last_price - entry) * size if signal == "COMPRAR" else (entry - last_price) * size
    return {"result": "CLOSE", "profit": profit}


# ---------------- BACKTEST ----------------
def backtest_symbol(symbol):
    df = download_klines(symbol)
    trades = []
    capital = CAPITAL_INICIAL

    logging.info(f"🏁 Iniciando backtest en {symbol} ({len(df)} velas)")

    for i in range(100, len(df) - LOOKAHEAD):
        sub_df = df.iloc[:i + 1]
        signal_info = compute_signal_for_symbol_backtest(symbol, balance=capital, df_feat=sub_df)

        signal = signal_info.get("signal")
        entry = signal_info.get("entry_suggest")
        sl = signal_info.get("SL")
        tp = signal_info.get("TP")

        # 🔸 Validación mínima
        if signal not in ["COMPRAR", "VENTA"]:
            continue
        if any(isnan(x) for x in [entry, sl, tp]):
            continue

        # 🧮 Posición según riesgo
        risk_amount = capital * RISK_PER_TRADE
        stop_distance = abs(entry - sl)
        if stop_distance == 0:
            continue
        size = risk_amount / stop_distance

        future_df = df.iloc[i + 1:i + LOOKAHEAD]
        resultado = simular_trade(entry, sl, tp, future_df, size, signal)
        capital += resultado["profit"]

        trades.append({
            "symbol": symbol,
            "index": i,
            "signal": signal,
            "entry_price": entry,
            "SL": sl,
            "TP": tp,
            "profit": resultado["profit"],
            "result": resultado["result"],
            "capital": capital,
            "pattern": signal_info.get("candle_pattern"),
            "support": signal_info.get("support"),
            "resistance": signal_info.get("resistance")
        })

    return trades


# ---------------- MÉTRICAS ----------------
def calcular_metricas(trades):
    df = pd.DataFrame(trades)
    total = len(df)
    if total == 0:
        return {"total": 0, "winrate": 0, "pf": 0, "capital_final": CAPITAL_INICIAL}

    wins = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]
    winrate = (len(wins) / total) * 100
    pf = wins["profit"].sum() / abs(losses["profit"].sum()) if len(losses) > 0 else np.inf
    capital_final = df["capital"].iloc[-1]
    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(winrate, 2),
        "pf": round(pf, 2),
        "capital_final": round(capital_final, 2)
    }


# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_trades = []
    for s in SYMBOLS:
        t = backtest_symbol(s)
        all_trades.extend(t)

    df_all = pd.DataFrame(all_trades)
    df_all.to_csv("backtest_resultados_completos.csv", index=False)

    metrics = calcular_metricas(all_trades)
    print("📊 === RESULTADOS GLOBALES ===")
    print(f"Total trades: {metrics['total']}")
    print(f"✅ Ganadores: {metrics['wins']} ❌ Perdidos: {metrics['losses']}")
    print(f"🏆 Winrate: {metrics['winrate']}%")
    print(f"⚖️ Profit Factor: {metrics['pf']}")
    print(f"💰 Capital final: ${metrics['capital_final']}")

    logging.info("✅ Backtest completo — estrategia IA + SR + patrones + SL/TP")
