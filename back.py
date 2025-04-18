import numpy as np
import tensorflow as tf
import cv2
import logging
import json
from flask import Flask, request, jsonify, render_template

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
port = 4000

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('saved_modelN2.keras')

# Cargar el mapa de etiquetas desde label_map.json
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

# Invertir el mapa de etiquetas para obtener la asignación de índice a etiqueta
label_map = {v: k for k, v in label_map.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug(f"Solicitud recibida con encabezados: {request.headers}")
    logging.debug(f"Archivos recibidos en la solicitud: {request.files.keys()}")
    
    if 'image' not in request.files or request.files['image'].filename == '':
        logging.error("No se subió una imagen válida")
        return jsonify({'error': 'No se subió una imagen válida'}), 400
    
    file = request.files['image']
    logging.debug(f"Archivo recibido: {file.filename}, Tipo de contenido: {file.content_type}")
    
    try:
        # Leer y decodificar la imagen
        image = np.frombuffer(file.read(), dtype=np.uint8)
        uploaded_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if uploaded_image is None:
            logging.error("No se pudo decodificar la imagen. Posiblemente un formato no soportado.")
            return jsonify({'error': 'Formato de imagen no válido'}), 400
        
        # Preprocesar la imagen: redimensionar y normalizar
        uploaded_image = cv2.resize(uploaded_image, (200, 200)) / 255.0  # Redimensionar y normalizar
        uploaded_image = np.expand_dims(uploaded_image, axis=0)  # Ajustar dimensiones para el modelo
        
        # Hacer la predicción
        prediction = model.predict(uploaded_image)
        predicted_label = int(np.argmax(prediction))  # Obtener el índice de la clase predicha
        probability = float(np.max(prediction))  # Obtener la probabilidad de la predicción
        
        # Obtener la etiqueta legible desde el mapa de etiquetas
        predicted_label_name = label_map.get(predicted_label, 'Desconocido')
        
        logging.info(f"Etiqueta Predicha: {predicted_label_name}, Probabilidad: {probability}")
        return jsonify({'etiqueta_predicha': predicted_label_name, 'probabilidad': probability}), 200
    except Exception as e:
        logging.exception("Error al procesar la imagen")
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
