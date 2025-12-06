from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import urllib.request
import os

app = Flask(__name__)

# -----------------------------------------------------------
# CONFIGURACIÓN: nombres de archivos de los modelos (ajusta rutas si hace falta)
# -----------------------------------------------------------
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Tamaño de entrada esperado por los modelos age/gender
MODEL_INPUT_SIZE = (227, 227)
MODEL_MEAN_VALUES = (78.426, 87.769, 114.896)  # medias en BGR para estos modelos Caffe

# -----------------------------------------------------------
# UTIL: comprobar que existen los archivos de modelo
# -----------------------------------------------------------
missing = [p for p in (AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL) if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(f"Faltan archivos de modelo: {', '.join(missing)}. "
                            "Coloca los archivos .caffemodel/.prototxt en la carpeta de la aplicación o ajusta las rutas.")

# -----------------------------------------------------------
# CARGA DE REDES (una sola vez al inicio)
# -----------------------------------------------------------
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

age_ranges = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]
gender_labels = ["Hombre", "Mujer"]

# Detector de rostros Haar Cascade (rápido, aunque menos preciso que detectores DNN)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------------------------------------
# FUNCIÓN: cargar imagen desde URL
# -----------------------------------------------------------
def load_image_from_url(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        img_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# -----------------------------------------------------------
# FUNCIÓN: preparar cara para el modelo (rescale manteniendo aspecto + padding)
# -----------------------------------------------------------
def prepare_face(face_bgr, size=MODEL_INPUT_SIZE, mean=MODEL_MEAN_VALUES):
    h, w = face_bgr.shape[:2]
    target_w, target_h = size
    # Escalar manteniendo relación de aspecto
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(face_bgr, (nw, nh))
    # Imagen rellenada (padding) con fondo negro
    padded = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    padded[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    blob = cv2.dnn.blobFromImage(padded, 1.0, size, mean, swapRB=False, crop=False)
    return blob

# -----------------------------------------------------------
# FUNCIÓN PRINCIPAL: analizar imagen (detectar rostros, predecir edad/género)
# -----------------------------------------------------------
def analyze_image(img):
    if img is None:
        return {"error": "Imagen vacía o inválida"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        # Ampliar un poco el bounding box para incluir contexto
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        face = img[y1:y2, x1:x2]

        # Preparar blob y ejecutar modelos de género y edad
        blob = prepare_face(face)

        # Género
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_idx = int(gender_preds[0].argmax())
        gender_label = gender_labels[gender_idx]
        gender_conf = float(gender_preds[0][gender_idx])

        # Edad
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_idx = int(age_preds[0].argmax())
        age_label = age_ranges[age_idx]
        age_conf = float(age_preds[0][age_idx])

        results.append({
            "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            "genero": {"etiqueta": gender_label, "confianza": round(gender_conf, 4)},
            "edad": {"etiqueta": age_label, "confianza": round(age_conf, 4)}
        })

    return {"caras": results, "cantidad": len(results)}

# -----------------------------------------------------------
# RUTAS
# -----------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # Página simple de índice (puedes reemplazar por una plantilla real)
    return (
        "<h2>API de detección de Edad/Género</h2>"
        "<p>Envía por POST una URL de imagen (campo de formulario 'url') o sube un archivo (campo 'image') a /analyze</p>"
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    # Acepta URL en form-data o archivo multipart 'image'
    img = None
    url = request.form.get("url")
    if url:
        img = load_image_from_url(url)
        if img is None:
            return jsonify({"error": "No se pudo cargar la imagen desde la URL proporcionada"}), 400

    elif 'image' in request.files:
        file = request.files['image']
        data = file.read()
        img_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "No se pudo decodificar la imagen subida"}), 400
    else:
        return jsonify({"error": "No se recibió 'url' ni archivo 'image' en la petición"}), 400

    try:
        results = analyze_image(img)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": "Error interno al procesar la imagen", "detalles": str(e)}), 500

# -----------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------
if __name__ == "__main__":
    # Para desarrollo local. En producción usa gunicorn/uWSGI detrás de un servidor.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
