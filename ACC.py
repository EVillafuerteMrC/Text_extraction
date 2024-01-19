import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances


def cargar_imagenes(ruta, target_size=(400, 600)):
    imagenes = []
    nombres = []
    for archivo in os.listdir(ruta):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_completa = os.path.join(ruta, archivo)
            imagen = cv2.imread(ruta_completa)
            # Redimensionar la imagen al tamaño deseado (ajustar target_size según tus necesidades)
            imagen = cv2.resize(imagen, target_size)
            imagenes.append(imagen)
            nombres.append(archivo)
    return np.array(imagenes), nombres

def extraer_caracteristicas(modelo, imagenes):
    imagenes_preprocesadas = preprocess_input(imagenes.copy())
    caracteristicas = modelo.predict(imagenes_preprocesadas)
    return caracteristicas

# Ruta a tu conjunto local de imágenes
ruta_conjunto = r"C:\Users\EVillafuerte\Documents\IA PY\Fotos_vertex\images\train"

# Cargar imágenes del conjunto local con redimensionamiento
conjunto_imagenes, nombres_conjunto = cargar_imagenes(ruta_conjunto)

# Cargar un modelo preentrenado (VGG16 en este caso)
modelo_base = VGG16(weights='imagenet', include_top=False)

# Ruta a tu imagen de referencia
ruta_referencia = "Fotos_vertex/images/train/Transaccion_PCH5.jpg"
referencia = cv2.imread(ruta_referencia)
# Redimensionar la imagen de referencia al tamaño deseado
referencia = cv2.resize(referencia, (400, 600))
referencia = np.expand_dims(referencia, axis=0)

# Extraer características de la imagen de referencia
caracteristicas_referencia = extraer_caracteristicas(modelo_base, referencia)
caracteristicas_referencia = caracteristicas_referencia.reshape((caracteristicas_referencia.shape[0], -1))

# Extraer características del conjunto local
caracteristicas_conjunto = extraer_caracteristicas(modelo_base, conjunto_imagenes)
caracteristicas_conjunto = caracteristicas_conjunto.reshape((caracteristicas_conjunto.shape[0], -1))

# Calcular similitud y encontrar la imagen más parecida
distancias = euclidean_distances(caracteristicas_referencia, caracteristicas_conjunto)
indice_similitud = np.argmin(distancias)
porcentaje_coincidencia = 100 * (1 - distancias.min() / distancias.max())
imagen_similar = nombres_conjunto[indice_similitud]

def encontrar_rectangulos_verdes(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([60, 40, 40])
    verde_alto = np.array([80, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    puntos_referencia = []
    for contorno in contornos:
        x, y, ancho, alto = cv2.boundingRect(contorno)
        centro_x = x + ancho // 2
        centro_y = y + alto // 2
        puntos_referencia.append((centro_x, centro_y, ancho, alto))

    return puntos_referencia

def transferir_rectangulos(imagen_origen, imagen_destino, puntos_referencia):
    for punto in puntos_referencia:
        x, y, ancho, alto = punto

        # crear variaciones para los recibos y cheques
        x_destino = x 
        y_destino = y 

        # Dibujar el rectángulo en la imagen destino
        cv2.rectangle(imagen_destino, (x_destino - ancho // 2, y_destino - alto // 2),
                      (x_destino + ancho // 2, y_destino + alto // 2), (0, 255, 0), 2)

    return imagen_destino

# Cargar las imágenes

# Encontrar rectángulos verdes en la imagen con rectángulos


if porcentaje_coincidencia >= 85:
    print(f"La imagen más similar es: {imagen_similar} con un {porcentaje_coincidencia:.2f}% de coincidencia.")

    imagen_similar = cv2.imread("Fotos_vertex/images/labelsimg/"+imagen_similar)
    ruta_referencia = cv2.imread(ruta_referencia)

    imagen_similar = cv2.resize(imagen_similar, (400, 600))
    ruta_referencia = cv2.resize(ruta_referencia, (400, 600))

    puntos_referencia = encontrar_rectangulos_verdes(imagen_similar)
    resultado = transferir_rectangulos(imagen_similar.copy(), ruta_referencia.copy(), puntos_referencia)  
    cv2.imshow('Imagen Resultante', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No se encontró una coincidencia con suficiente porcentaje. Porcentaje de coincidencia: {porcentaje_coincidencia:.2f}% con la imagen {imagen_similar}")


