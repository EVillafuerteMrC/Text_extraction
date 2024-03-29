import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw
import io

# Ruta a tu conjunto local de imágenes
ruta_conjunto = r"C:\Users\EVillafuerte\Documents\IA PY\Fotos_vertex\images\train"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\EVillafuerte\Documents\IA PY\mimetic-scion-407221-93093bee51a5.json"
imagen_a_analizar = "Fotos_vertex/images/train/Transaccion_PCH2.jpg"



def cargar_imagenes(ruta, target_size=(400, 600)):
    imagenes = []
    nombres = []
    for archivo in os.listdir(ruta):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_completa = os.path.join(ruta, archivo)
            imagen = cv2.imread(ruta_completa)
            imagen = cv2.resize(imagen, target_size)
            imagenes.append(imagen)
            nombres.append(archivo)
    return np.array(imagenes), nombres



def extraer_caracteristicas(modelo, imagenes):
    imagenes_preprocesadas = preprocess_input(imagenes.copy())
    caracteristicas = modelo.predict(imagenes_preprocesadas)
    return caracteristicas



conjunto_imagenes, nombres_conjunto = cargar_imagenes(ruta_conjunto)

# Cargar un modelo preentrenado (VGG16 en este caso)
modelo_base = VGG16(weights='imagenet', include_top=False)


referencia = cv2.imread(imagen_a_analizar)
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

def extraer_texto_google_vision_en_rectangulos_verdes(ruta_imagen, puntos_referencia):
    client = vision_v1.ImageAnnotatorClient()

    with io.open(ruta_imagen, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    resultados_texto = []

    for punto in puntos_referencia:
        x, y, ancho, alto = punto
        texto_rectangulo = []

        # Verificar si el centro del texto está dentro de algún rectángulo verde
        for text in texts:
            centro_texto_x = (text.bounding_poly.vertices[0].x + text.bounding_poly.vertices[2].x) // 2
            centro_texto_y = (text.bounding_poly.vertices[0].y + text.bounding_poly.vertices[2].y) // 2

            if x - ancho // 2 <= centro_texto_x <= x + ancho // 2 and y - alto // 2 <= centro_texto_y <= y + alto // 2:
                texto_rectangulo.append(text.description)

        # Agregar el texto del rectángulo actual a los resultados
        resultados_texto.append(" ".join(texto_rectangulo))
    os.remove('TempImgFinal.png')
    return resultados_texto
    


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


if porcentaje_coincidencia >= 85:
    print(f"La imagen más similar es: {imagen_similar} con un {porcentaje_coincidencia:.2f}% de coincidencia.")

    imagen_similar = cv2.imread("Fotos_vertex/images/labelsimg/"+imagen_similar)
    imagen_a_analizar = cv2.imread(imagen_a_analizar)

    imagen_similar = cv2.resize(imagen_similar, (400, 600))
    imagen_a_analizar = cv2.resize(imagen_a_analizar, (400, 600))

    puntos_referencia = encontrar_rectangulos_verdes(imagen_similar)
    resultado = transferir_rectangulos(imagen_similar.copy(), imagen_a_analizar.copy(), puntos_referencia)  
    cv2.imwrite('TempImgFinal.png', resultado)
    cv2.imshow('Imagen Resultante', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    texto_extraido = extraer_texto_google_vision_en_rectangulos_verdes('TempImgFinal.png', puntos_referencia)

    # Imprimir el texto extraído
    print("Texto extraído de los rectángulos verdes en la imagen TempImgFinal:")
    for i, texto in enumerate(texto_extraido):
        if(texto):
            print(f"{i+1}. {texto}")

else:
    print(f"No se encontró una coincidencia con suficiente porcentaje. Porcentaje de coincidencia: {porcentaje_coincidencia:.2f}% con la imagen {imagen_similar}")


