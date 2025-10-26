import logging
import warnings
import pandas as pd
from binance.client import Client
import bot
from bot import init_client

# ==============================
# ⚠️ Configuración general
# ==============================
warnings.filterwarnings("ignore", category=FutureWarning)  # 🔇 para sklearn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

API_KEY = "TU_API_KEY"
API_SECRET = "TU_API_SECRET"

# 💰 Configuración de capital
CAPITAL_INICIAL = 10000  # USD
RIESGO_POR_TRADE = 0.01  # 1%

# ==============================
# 🔐 Inicializar cliente global
# ==============================
init_client(API_KEY, API_SECRET)

# ==============================
# 📊 Lista de símbolos a backtestear
# ==============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
DAYS = 90

# Lista para guardar métricas globales
global_results = []

# ==============================
# 🚀 Loop de backtest
# ==============================
for sym in SYMBOLS:
    logging.info(f"🚀 Probando estrategia para {sym} en {DAYS} días...")
    result = bot.backtest_symbol(sym, days=DAYS, use_planB=True)

    # ⚠️ Si hubo error
    if "error" in result:
        logging.warning(f"⚠️ {sym}: {result['error']}")
        continue

    # Extraer métricas
    trades = result.get("trades", 0)
    cum_r = result.get("cum_R", 0)
    winrate = result.get("winrate", 0)
    avgR = result.get("avgR", 0)
    maxdd = result.get("maxDD_R", 0)
    details = result.get("details", [])

    # Calcular ganadoras vs perdedoras
    wins = sum(1 for r in details if r["R"] > 0)
    losses = sum(1 for r in details if r["R"] <= 0)

    # 💰 Cálculo de ganancia en USD
    ganancia_usd = cum_r * (CAPITAL_INICIAL * RIESGO_POR_TRADE)
    capital_final = CAPITAL_INICIAL + ganancia_usd

    # 📊 Log detallado
    logging.info(f"📈 Total trades: {trades}")
    logging.info(f"✅ Ganadoras: {wins} | ❌ Perdedoras: {losses}")
    logging.info(f"🏆 Winrate: {winrate*100:.2f}%")
    logging.info(f"💵 Rentabilidad total: {cum_r:.2f} R")
    logging.info(f"📉 Max Drawdown: {maxdd:.2f} R")
    logging.info(f"💰 Ganancia estimada: ${ganancia_usd:,.2f} USD")
    logging.info(f"🏦 Capital final estimado: ${capital_final:,.2f} USD")

    # Guardar resultado resumen
    global_results.append({
        "symbol": sym,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate*100, 2),
        "cum_R": round(cum_r, 2),
        "avg_R": round(avgR, 2) if avgR is not None else None,
        "maxDD_R": round(maxdd, 2),
        "ganancia_USD": round(ganancia_usd, 2),
        "capital_final_USD": round(capital_final, 2)
    })

    # 📁 Guardar trades individuales si quieres analizar más
    df_trades = pd.DataFrame(details)
    df_trades.to_csv(f"backtest_{sym}.csv", index=False)
    logging.info(f"📊 Detalles de trades guardados en backtest_{sym}.csv")

# ==============================
# 📝 Exportar resumen global
# ==============================
if global_results:
    df_global = pd.DataFrame(global_results)
    df_global.to_csv("backtest_resumen.csv", index=False)
    logging.info("✅ Resumen global guardado en backtest_resumen.csv")

    logging.info("\n📊 RESULTADOS FINALES:")
    logging.info(df_global.to_string(index=False))
else:
    logging.warning("⚠️ No se generaron resultados globales.")



