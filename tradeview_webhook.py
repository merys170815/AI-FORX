from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Carga tu modelo entrenado
model = load("fx_model_ai_pro.pkl")

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    # Espera un JSON con Open, High, Low, Close
    df = pd.DataFrame([data])

    # Aquí se calculan las mismas features que en tu script
    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()
    df["rsi14"] = df["Close"].pct_change().rolling(14).apply(lambda x: (x[x > 0].sum() / abs(x).sum()) * 100)
    # ... añade todas las features necesarias como en tu script original

    # Predicción
    X = df[["ema20", "ema50", "ema200", "rsi14"]]  # reemplaza con todas tus features
    p_up = model.predict_proba(X)[0, 1]

    signal = "LONG" if p_up > 0.65 else "SHORT" if p_up < 0.35 else "NEUTRAL"
    return jsonify({"signal": signal, "prob_up": float(p_up)})


if __name__ == "__main__":
    app.run(port=5000)

    # Forzar redeploy en Railway

