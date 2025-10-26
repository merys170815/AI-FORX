# ===============================
# 🧠 adaptive_learning.py — Aprendizaje Adaptativo Diario + Multi-Horizonte
# ===============================
import json
import time
import threading
import schedule
import logging
import os
from datetime import datetime
from trade_engine import get_stats   # 👈 Ya lo tienes en tu proyecto

# 👉 Archivos de configuración
THRESHOLD_STATE_FILE = "thresholds_state.json"
MAX_THRESHOLD = 0.80
MIN_THRESHOLD = 0.40

# 👉 Filtros globales dinámicos (pueden sincronizarse con app.py)
filtros_config = {
    "min_pf": 1.5,
    "min_accuracy": 50.0,
    "max_drawdown": 0.20
}

# 👉 Lista de símbolos activos
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]

# 👉 Definir horizontes de estrategia
# Cada horizonte tendrá sus propios thresholds adaptativos
HORIZONTES = ["scalping", "intraday", "swing"]

# ===============================
# 🧾 Funciones básicas de estado
# ===============================
def _ensure_thresholds_file():
    """Garantiza que el archivo thresholds_state.json exista."""
    if not os.path.exists(THRESHOLD_STATE_FILE):
        with open(THRESHOLD_STATE_FILE, "w") as f:
            json.dump({"history": {}, "thresholds": {}}, f, indent=2)

def _read_state():
    _ensure_thresholds_file()
    with open(THRESHOLD_STATE_FILE, "r") as f:
        return json.load(f)

def _write_state(data):
    with open(THRESHOLD_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===============================
# 🧠 Aprendizaje Adaptativo Diario
# ===============================
def daily_adaptive_learning():
    """🧠 Ajusta umbrales y filtros automáticamente según el rendimiento diario."""
    try:
        logging.info("🧠 [Learning] Iniciando aprendizaje adaptativo diario...")

        # 1️⃣ Obtener métricas de rendimiento (ya lo tienes en trade_engine)
        stats = get_stats()
        winrate = stats.get("winrate", 0)
        avg_rr = stats.get("avg_rr", 0)
        pf_by_symbol = stats.get("pf_by_symbol", {})

        # Opcional: si en un futuro agregas pf por horizonte
        pf_by_symbol_horizon = stats.get("pf_by_symbol_horizon", {})

        state = _read_state()

        # ===============================
        # 2️⃣ Ajuste global de filtros
        # ===============================
        if winrate > 60:
            filtros_config["min_pf"] = min(filtros_config["min_pf"] + 0.05, 2.5)
        elif winrate < 45:
            filtros_config["min_pf"] = max(filtros_config["min_pf"] - 0.05, 1.0)

        # ===============================
        # 3️⃣ Ajuste thresholds por símbolo y horizonte
        # ===============================
        for sym in SYMBOLS:
            # Inicializar estructura si no existe
            if sym not in state["thresholds"]:
                state["thresholds"][sym] = {}

            for h in HORIZONTES:
                current = state["thresholds"][sym].get(h, {"UP": 0.55, "DOWN": 0.55})

                # PF general si no hay específico por horizonte
                pf = pf_by_symbol_horizon.get(sym, {}).get(h, pf_by_symbol.get(sym, 1.0))

                if pf > 1.3:
                    current["UP"] = min(current["UP"] + 0.02, MAX_THRESHOLD)
                    current["DOWN"] = min(current["DOWN"] + 0.02, MAX_THRESHOLD)
                elif pf < 0.9:
                    current["UP"] = max(current["UP"] - 0.02, MIN_THRESHOLD)
                    current["DOWN"] = max(current["DOWN"] - 0.02, MIN_THRESHOLD)

                state["thresholds"][sym][h] = current

        # ===============================
        # 4️⃣ Guardar histórico diario
        # ===============================
        state.setdefault("history", {})[str(datetime.utcnow().date())] = {
            "winrate": winrate,
            "avg_rr": avg_rr,
            "filtros": filtros_config.copy(),
            "pf_by_symbol": pf_by_symbol,
            "pf_by_symbol_horizon": pf_by_symbol_horizon
        }

        _write_state(state)
        logging.info("✅ [Learning] Thresholds y filtros adaptados con éxito")

    except Exception as e:
        logging.error(f"❌ [Learning] Error en aprendizaje adaptativo diario: {e}")

# ===============================
# ⏳ Scheduler diario
# ===============================
def schedule_learning():
    """⏳ Programa la tarea diaria a medianoche UTC."""
    schedule.every().day.at("00:00").do(daily_adaptive_learning)
    while True:
        schedule.run_pending()
        time.sleep(60)

# ===============================
# 🚀 Inicializador desde app.py
# ===============================
def start_learning_thread():
    t = threading.Thread(target=schedule_learning, daemon=True)
    t.start()
    logging.info("🧠 Aprendizaje adaptativo diario (multi-horizonte) activado.")

