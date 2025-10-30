# ==============================================
# 🔍 AI IMAGE DETECTIVE - BACKEND (Flask)
# Desarrollado para detección de imágenes generadas por IA
# ==============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# ==============================================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================================
app = Flask(__name__)
CORS(app)  # Permitir conexión desde el frontend (React)

# Ruta del modelo entrenado
MODEL_PATH = "saved_model/modelo_cnn_final.h5"

# Si la carpeta no existe, crearla
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ==============================================
# 🧠 CARGA DEL MODELO
# ==============================================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente desde:", MODEL_PATH)
except Exception as e:
    print("⚠️ Error al cargar el modelo:", e)
    model = None


# ==============================================
# 🧠 FUNCIÓN DE PREDICCIÓN (ajustada y calibrada)
# ==============================================
def predict_image(img_bytes, thresh_override=None):
    from PIL import Image, ImageOps
    import numpy as np
    import io

    # --- 1) Cargar y preparar base (260x260) ---
    base = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((260, 260))

    def prep(pil_img):
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0)

    # --- 2) TTA: original + espejo + 5-crops (centro y esquinas) ---
    W = H = 260
    crop = 232  # recorte leve (≈ 10%) para eliminar bordes raros y reescalar
    coords = [
        (0, 0, crop, crop),                             # sup-izq
        (W - crop, 0, W, crop),                        # sup-der
        (0, H - crop, crop, H),                        # inf-izq
        (W - crop, H - crop, W, H),                    # inf-der
        ((W - crop)//2, (H - crop)//2, (W + crop)//2, (H + crop)//2),  # centro
    ]

    variants = [base, ImageOps.mirror(base)]
    for c in coords:
        variants.append(base.crop(c).resize((260, 260), Image.BILINEAR))

    X = [prep(v) for v in variants]

    # --- 3) Promediar predicciones: prob_real = P(y=1=REAL) ---
    probs = [float(model.predict(x, verbose=0)[0][0]) for x in X]
    prob_real = float(np.mean(probs))

    # --- 4) Umbral: más permisivo con REAL (tune rápido) ---
    THRESH = 0.25
    if isinstance(thresh_override, (int, float)):
        THRESH = float(thresh_override)

    if prob_real >= THRESH:
        clase = "REAL"; confianza = prob_real
    else:
        clase = "IA";   confianza = 1.0 - prob_real

    # --- 5) Log útil para depurar en consola ---
    print(f"[PRED] prob_real={prob_real:.4f}  THRESH={THRESH:.2f}  -> {clase}")

    return {
        "resultado": clase,
        "confianza": round(confianza, 4),
        "prob_real": round(prob_real, 4)
    }


# ==============================================
# 📤 ENDPOINT PRINCIPAL
# ==============================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para recibir una imagen y devolver la predicción.
    """
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No se ha subido ninguna imagen"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        result = predict_image(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================
# 🚀 INICIO DEL SERVIDOR FLASK
# ==============================================
if __name__ == "__main__":
    print("🚀 Servidor Flask iniciado en http://127.0.0.1:5000")
    print("Esperando peticiones desde el frontend...")
    app.run(host="0.0.0.0", port=5000, debug=True)
