import os
from binance.um_futures import UMFutures

# 🔑 Si quieres usar variables de entorno, descomenta esto y configura tus claves en el sistema:
# API_KEY = os.getenv("API_KEY")
# API_SECRET = os.getenv("API_SECRET")

# 🔑 O simplemente pégalos directamente aquí:
API_KEY = "TU_API_KEY_AQUI"
API_SECRET = "TU_SECRET_KEY_AQUI"

# Testnet de Binance Futures
base_url = "https://testnet.binancefuture.com"

client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=base_url)

print("🔍 Probando conexión con tus claves...")

try:
    balance = client.balance()
    print("✅ Conexión exitosa. Tu balance de prueba es:")
    for b in balance:
        print(f"{b['asset']}: {b['balance']}")
except Exception as e:
    print("❌ Error al conectar:", e)
