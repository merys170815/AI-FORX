from logger_trades import log_trade

symbol = "ETHUSDT"
signal = "COMPRA CONFIRMADA ✅"
entry_price = 3500.25
exit_price = 3520.80
pnl_usd = (exit_price - entry_price) * 0.1
pnl_pct = (exit_price - entry_price) / entry_price * 100
prob_up = 82.5
prob_down = 17.5
narrative = "Ruptura de resistencia con volumen alto."

log_trade(symbol, signal, entry_price, exit_price, pnl_usd, pnl_pct, prob_up, prob_down, signal, narrative)
