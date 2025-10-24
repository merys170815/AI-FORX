# trade_engine.py
import time
import logging
from datetime import datetime
from logger_trades import log_trade

# 📊 Configuración inicial
VIRTUAL_BALANCE = 10000.0
OPEN_TRADES = []
CLOSED_TRADES = []

def open_trade(symbol, signal, entry_price, sl, tp, size_usd=100):
    trade = {
        "symbol": symbol,
        "signal": signal,
        "entry_price": entry_price,
        "sl": sl,
        "tp": tp,
        "size_usd": size_usd,
        "open_time": datetime.utcnow(),
        "status": "OPEN"
    }
    OPEN_TRADES.append(trade)
    logging.info(f"🚀 Trade simulado abierto: {symbol} {signal} @ {entry_price} TP={tp} SL={sl}")

def simulate_market_tick(current_prices: dict):
    """
    current_prices: dict { "BTCUSDT": 65000.0, "ETHUSDT": 3000.0 }
    Se llama en cada ciclo para ver si TP o SL fueron alcanzados.
    """
    global VIRTUAL_BALANCE
    for trade in list(OPEN_TRADES):
        symbol = trade["symbol"]
        if symbol not in current_prices:
            continue
        price_now = current_prices[symbol]
        entry = trade["entry_price"]
        tp = trade["tp"]
        sl = trade["sl"]
        size = trade["size_usd"]
        direction = "long" if "COMPRA" in trade["signal"] else "short"

        pnl = 0
        hit = None

        if direction == "long":
            if price_now >= tp:
                pnl = (tp - entry) / entry * size
                hit = "TP"
            elif price_now <= sl:
                pnl = (sl - entry) / entry * size
                hit = "SL"
        else:
            if price_now <= tp:
                pnl = (entry - tp) / entry * size
                hit = "TP"
            elif price_now >= sl:
                pnl = (entry - sl) / entry * size
                hit = "SL"

        if hit:
            trade["status"] = "CLOSED"
            trade["close_price"] = price_now
            trade["close_time"] = datetime.utcnow()
            trade["pnl_usd"] = round(pnl, 2)
            VIRTUAL_BALANCE += pnl
            CLOSED_TRADES.append(trade)
            OPEN_TRADES.remove(trade)
            logging.info(f"✅ Trade {symbol} cerrado por {hit} | PnL: {pnl:.2f} USD | Balance: {VIRTUAL_BALANCE:.2f}")

            log_trade(
                symbol,
                trade["signal"],
                trade["entry_price"],
                price_now,
                pnl,
                (pnl / size) * 100,
                0, 0,
                f"CERRADO {hit}",
                f"Trade cerrado en {hit}"
            )

def get_stats():
    total = len(CLOSED_TRADES)
    wins = sum(1 for t in CLOSED_TRADES if t["pnl_usd"] > 0)
    losses = total - wins
    winrate = (wins / total * 100) if total > 0 else 0
    profit = sum(t["pnl_usd"] for t in CLOSED_TRADES)
    return {
        "balance": round(VIRTUAL_BALANCE, 2),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 2),
        "profit_total": round(profit, 2)
    }
