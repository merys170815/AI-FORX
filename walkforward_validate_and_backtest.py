import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import classification_report
import argparse
import os

# ------------------------------------------------------------
# ⚙️ CONFIGURACIÓN
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="binance_ai_lgbm_optuna_multi_v2_multiclass.pkl")
parser.add_argument("--csv", default="binance_predictions_optuna_multi_multiclass.csv")
parser.add_argument("--window", type=int, default=400)
parser.add_argument("--step", type=int, default=100)
parser.add_argument("--capital", type=float, default=10000.0)
parser.add_argument("--pos_frac", type=float, default=0.1)
args = parser.parse_args()

MODEL_FILE = args.model
CSV_FILE = args.csv
WINDOW = args.window
STEP = args.step
CAPITAL_INIT = args.capital
POS_FRAC = args.pos_frac

print(f"⚙️ Mode: pretrained | Model: {MODEL_FILE} | CSV: {CSV_FILE}")
print(f"⚙️ Window={WINDOW}, Step={STEP}, start_capital={CAPITAL_INIT}, pos_frac={POS_FRAC}")

# ------------------------------------------------------------
# 🧠 CARGA MODELO Y DATASET
# ------------------------------------------------------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Modelo no encontrado: {MODEL_FILE}")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Archivo CSV no encontrado: {CSV_FILE}")

model_bundle = load(MODEL_FILE)
if isinstance(model_bundle, dict):
    model = model_bundle.get("model", None)
    features = model_bundle.get("features", None)
else:
    model = model_bundle
    features = None

if model is None:
    raise ValueError("El modelo cargado no contiene un objeto válido.")

df = pd.read_csv(CSV_FILE)
if "symbol" not in df.columns or "target" not in df.columns:
    raise ValueError("El CSV debe contener columnas 'symbol' y 'target'.")

symbols = df["symbol"].unique()
print(f"✅ Dataset cargado con {len(df)} filas y {len(symbols)} símbolos: {symbols}")

# ------------------------------------------------------------
# ⚡ SINCRONIZAR FEATURES CON EL MODELO
# ------------------------------------------------------------
if features is not None:
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"⚠️ Faltan {len(missing_feats)} features respecto al modelo: {missing_feats}")
        # Creamos columnas faltantes con 0 (para compatibilidad)
        for f in missing_feats:
            df[f] = 0.0
    df = df[features + ["symbol", "target"]]

# ------------------------------------------------------------
# ⚡ WALK-FORWARD + BACKTEST
# ------------------------------------------------------------
results = []
for sym in symbols:
    df_sym = df[df["symbol"] == sym].copy()
    df_sym = df_sym.dropna(subset=["target"])
    y_true_all, y_pred_all = [], []
    capital = CAPITAL_INIT

    print(f"\n🔹 Evaluando walk-forward para {sym}...")

    for start in range(0, len(df_sym) - WINDOW, STEP):
        end = start + WINDOW
        train = df_sym.iloc[start:end]
        test = df_sym.iloc[end:end + STEP]
        if len(test) == 0:
            break

        X_test = test.drop(columns=["target", "symbol"], errors="ignore")
        y_test = test["target"]

        try:
            preds = model.predict(X_test)
        except Exception as e:
            print(f"⚠️ Error al predecir en {sym}: {e}")
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

        # Backtesting básico
        for i, signal in enumerate(preds):
            ret = test.iloc[i]["logret"] if "logret" in test.columns else 0
            if signal == 1:
                capital *= (1 + POS_FRAC * ret)
            elif signal == -1:
                capital *= (1 - POS_FRAC * ret)

        print(f"  🔹 Evaluado bloque {start}→{end} | Capital: ${capital:.2f}")

    if len(y_true_all) > 0:
        print("\n", classification_report(y_true_all, y_pred_all, digits=4))
    else:
        print(f"⚠️ No hubo predicciones válidas para {sym}.")

    results.append({
        "symbol": sym,
        "final_capital": round(capital, 2),
        "samples": len(df_sym)
    })

# ------------------------------------------------------------
# 💾 GUARDAR RESULTADOS
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("walkforward_backtest_summary.csv", index=False)
print("\n✅ Backtest terminado. Resumen guardado en: walkforward_backtest_summary.csv")
print(results_df)
