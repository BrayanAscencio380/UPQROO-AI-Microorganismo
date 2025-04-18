import os 
import cv2
import random
import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tqdm import tqdm

img_size = 200 

def aumentar_imagen(imagen):
    if random.choice([True, False]):
        imagen = cv2.flip(imagen, 1)  
    if random.choice([True, False]):
        angulo = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((img_size//2, img_size//2), angulo, 1)
        imagen = cv2.warpAffine(imagen, M, (img_size, img_size))
    if random.choice([True, False]):
        valor = random.uniform(0.8, 1.2)  
        imagen = np.clip(imagen * valor, 0, 1)
    return imagen

def balancear_dataset(directorio_datos):
    conteo_carpetas = {}
    
    for carpeta in os.listdir(directorio_datos):
        ruta_carpeta = os.path.join(directorio_datos, carpeta)
        if os.path.isdir(ruta_carpeta):
            conteo_carpetas[carpeta] = len(os.listdir(ruta_carpeta))
    
    if not conteo_carpetas:
        print("No se encontraron carpetas válidas en el directorio del conjunto de datos.")
        return
    
    max_imagenes = max(conteo_carpetas.values())
    print("Máximo de imágenes en una carpeta:", max_imagenes)
    
    for carpeta, cantidad in conteo_carpetas.items():
        ruta_carpeta = os.path.join(directorio_datos, carpeta)
        imagenes = [cv2.imread(os.path.join(ruta_carpeta, img)) for img in os.listdir(ruta_carpeta)]
        imagenes = [cv2.resize(img, (img_size, img_size)) / 255.0 for img in imagenes if img is not None]

        if not imagenes:
            print(f"Se omite la carpeta '{carpeta}' porque no se encontraron imágenes válidas.")
            continue
        
        while cantidad < max_imagenes:
            imagen = random.choice(imagenes)
            nueva_imagen = aumentar_imagen(imagen)
            nuevo_nombre = os.path.join(ruta_carpeta, f"aug_{cantidad}.jpg")
            cv2.imwrite(nuevo_nombre, (nueva_imagen * 255).astype(np.uint8))
            cantidad += 1
        
        print(f"Se ha equilibrado la carpeta {carpeta} con {max_imagenes} imágenes.")

def cargar_y_procesar_datos(directorio_datos):
    imagenes, etiquetas = [], []
    caracteres = sorted(os.listdir(directorio_datos))
    mapa_caracteres = {car: idx for idx, car in enumerate(caracteres)}
    
    for caracter in caracteres:
        directorio_caracter = os.path.join(directorio_datos, caracter)
        if not os.path.isdir(directorio_caracter):
            continue
        archivos_imagen = os.listdir(directorio_caracter)
        
        for archivo in tqdm(archivos_imagen, desc=f"Procesando {caracter}"):
            ruta_imagen = os.path.join(directorio_caracter, archivo)
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                imagen = cv2.resize(imagen, (img_size, img_size)) / 255.0
                imagenes.append(imagen)
                etiquetas.append(mapa_caracteres[caracter])
            except Exception as e:
                print(f"Error al procesar {ruta_imagen}: {e}")
    
    return np.array(imagenes), np.array(etiquetas), mapa_caracteres

def construir_modelo(dimension_entrada, num_clases):
    modelo = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=dimension_entrada),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),        
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),        
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ])

    modelo.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    return modelo

def main():
    directorio_datos = os.path.join(os.path.dirname(__file__), 'OCR_dataset')
    directorio_entrenamiento = os.path.join(directorio_datos, 'data/training_data')

    balancear_dataset(directorio_entrenamiento)

    imagenes, etiquetas, mapa_etiquetas = cargar_y_procesar_datos(directorio_entrenamiento)
    if len(imagenes) == 0:
        raise ValueError("No se encontraron imágenes en el conjunto de datos.")

    print("\nMapa de etiquetas:")
    for nombre_clase, etiqueta in mapa_etiquetas.items():
        print(f"{etiqueta}: {nombre_clase}")

    with open("mapa_etiquetas.json", "w") as f:
        json.dump(mapa_etiquetas, f, indent=4)

    imagenes_entrenamiento, imagenes_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)
    imagenes_entrenamiento, imagenes_validacion, etiquetas_entrenamiento, etiquetas_validacion = train_test_split(imagenes_entrenamiento, etiquetas_entrenamiento, test_size=0.125, random_state=42)

    print("\nDimensiones de los datos:")
    print("Imágenes de entrenamiento:", imagenes_entrenamiento.shape)
    print("Etiquetas de entrenamiento:", etiquetas_entrenamiento.shape)
    print("Imágenes de validación:", imagenes_validacion.shape)
    print("Etiquetas de validación:", etiquetas_validacion.shape)
    print("Imágenes de prueba:", imagenes_prueba.shape)
    print("Etiquetas de prueba:", etiquetas_prueba.shape)

    num_clases = len(mapa_etiquetas)
    modelo = construir_modelo((img_size, img_size, 3), num_clases)
    modelo.summary()

    callbacks = [
        ModelCheckpoint("mejor_modelo.keras", monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    ]

    historial = modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento, 
                           validation_data=(imagenes_validacion, etiquetas_validacion), 
                           epochs=10, 
                           batch_size=32,
                           callbacks=callbacks)

    perdida_prueba, precision_prueba = modelo.evaluate(imagenes_prueba, etiquetas_prueba)
    print('Precisión en prueba:', precision_prueba)

    modelo.save('modelo_ia.keras')

if __name__ == "__main__":
    main()
