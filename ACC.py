# import os
# import cv2
# import numpy as np
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import euclidean_distances
# from google.cloud import vision_v1

# def cargar_imagenes(ruta, target_size=(400, 600)):
#     imagenes = []
#     nombres = []
#     for archivo in os.listdir(ruta):
#         if archivo.endswith(".jpg") or archivo.endswith(".png"):
#             ruta_completa = os.path.join(ruta, archivo)
#             imagen = cv2.imread(ruta_completa)
#             # Redimensionar la imagen al tamaño deseado (ajustar target_size según tus necesidades)
#             imagen = cv2.resize(imagen, target_size)
#             imagenes.append(imagen)
#             nombres.append(archivo)
#     return np.array(imagenes), nombres

# def extraer_caracteristicas(modelo, imagenes):
#     imagenes_preprocesadas = preprocess_input(imagenes.copy())
#     caracteristicas = modelo.predict(imagenes_preprocesadas)
#     return caracteristicas

# # Ruta a tu conjunto local de imágenes
# ruta_conjunto = r"C:\Users\EVillafuerte\Documents\IA PY\Fotos_vertex\images\train"

# # Cargar imágenes del conjunto local con redimensionamiento
# conjunto_imagenes, nombres_conjunto = cargar_imagenes(ruta_conjunto)

# # Cargar un modelo preentrenado (VGG16 en este caso)
# modelo_base = VGG16(weights='imagenet', include_top=False)

# # Ruta a tu imagen de referencia
# ruta_referencia = r"C:\Users\EVillafuerte\Documents\IA PY\Fotos_vertex\images\Transaccion_PCH8.jpg"
# referencia = cv2.imread(ruta_referencia)
# # Redimensionar la imagen de referencia al tamaño deseado
# referencia = cv2.resize(referencia, (400, 600))
# referencia = np.expand_dims(referencia, axis=0)

# # Extraer características de la imagen de referencia
# caracteristicas_referencia = extraer_caracteristicas(modelo_base, referencia)
# caracteristicas_referencia = caracteristicas_referencia.reshape((caracteristicas_referencia.shape[0], -1))

# # Extraer características del conjunto local
# caracteristicas_conjunto = extraer_caracteristicas(modelo_base, conjunto_imagenes)
# caracteristicas_conjunto = caracteristicas_conjunto.reshape((caracteristicas_conjunto.shape[0], -1))

# # Calcular similitud y encontrar la imagen más parecida
# distancias = euclidean_distances(caracteristicas_referencia, caracteristicas_conjunto)
# indice_similitud = np.argmin(distancias)
# porcentaje_coincidencia = 100 * (1 - distancias.min() / distancias.max())
# imagen_similar = nombres_conjunto[indice_similitud]




# def second_step (image_path_2,image_path_1):

#     base = r"C:/Users/EVillafuerte/Documents/IA PY/Fotos_vertex/images/labelsimg/"

#     image_path_1 = base + image_path_1
#     original_image_1 = cv2.imread(image_path_1)

#     # Convierte la imagen a formato HSV
#     hsv = cv2.cvtColor(original_image_1, cv2.COLOR_BGR2HSV)

#     # Define el rango de colores verdes en HSV
#     lower_green = np.array([30, 40, 40])
#     upper_green = np.array([80, 255, 255])

#     # Filtra los píxeles verdes
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Encuentra contornos en la máscara
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Lista para almacenar las coordenadas de los rectángulos
#     rectangles_coordinates = []

#     # Itera sobre los contornos para encontrar rectángulos
#     for contour in contours:
#         # Ajusta un polígono al contorno
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Si el polígono tiene cuatro vértices, es probable que sea un rectángulo
#         if len(approx) == 4:
#             # Obtiene las coordenadas del rectángulo
#             x, y, w, h = cv2.boundingRect(approx)

#             # Guarda las coordenadas en la lista
#             rectangles_coordinates.append((x, y, w, h))

#     # Carga la segunda imagen
#     original_image_2 = cv2.imread(image_path_2)

#     # Itera sobre las coordenadas de los rectángulos y dibuja en la segunda imagen
#     for (x, y, w, h) in rectangles_coordinates:
#         # Dibuja el rectángulo en la segunda imagen
#         cv2.rectangle(original_image_2, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Muestra la segunda imagen con los rectángulos dibujados
#     cv2.imshow('Rectángulos en Segunda Imagen', original_image_2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if porcentaje_coincidencia >= 85:
#     print(f"La imagen más similar es: {imagen_similar} con un {porcentaje_coincidencia:.2f}% de coincidencia.")
#     try:
#         second_step(ruta_referencia, imagen_similar)
#     except:
#         print("Error en el proceso de comprobación")
        
# else:
#     print(f"No se encontró una coincidencia con suficiente porcentaje. Porcentaje de coincidencia: {porcentaje_coincidencia:.2f}% con la imagen {imagen_similar}")





import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
from google.cloud import vision_v1
from google.cloud import vision
from google.cloud.vision_v1 import types
import base64

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\EVillafuerte\Documents\IA PY\mimetic-scion-407221-93093bee51a5.json"

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
ruta_referencia = r"C:\Users\EVillafuerte\Documents\IA PY\Fotos_vertex\images\Transaccion_PCH8.jpg"
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













def extract_text_from_rectangles():
    # Configura el cliente de Vision
    client = vision.ImageAnnotatorClient()
    image_path = 'FinalImage.jpg'
    # Lee la imagen
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Convierte la imagen a un objeto de tipo Image
    image = types.Image(content=content)

    # Realiza la solicitud para la detección de texto
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extrae el texto de cada rectángulo
    result = []
    for text in texts[1:]:  # El primer elemento es el texto completo de la imagen
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        result.append({
            'text': text.description,
            'vertices': vertices
        })

    return result




























def second_step (image_path_2,image_path_1):

    base = r"C:/Users/EVillafuerte/Documents/IA PY/Fotos_vertex/images/labelsimg/"

    image_path_1 = base + image_path_1
    original_image_1 = cv2.imread(image_path_1)

    # Convierte la imagen a formato HSV
    hsv = cv2.cvtColor(original_image_1, cv2.COLOR_BGR2HSV)

    # Define el rango de colores verdes en HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Filtra los píxeles verdes
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Encuentra contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las coordenadas de los rectángulos
    rectangles_coordinates = []

    # Itera sobre los contornos para encontrar rectángulos
    for contour in contours:
        # Ajusta un polígono al contorno
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el polígono tiene cuatro vértices, es probable que sea un rectángulo
        if len(approx) == 4:
            # Obtiene las coordenadas del rectángulo
            x, y, w, h = cv2.boundingRect(approx)

            # Guarda las coordenadas en la lista
            rectangles_coordinates.append((x, y, w, h))

    # Carga la segunda imagen
    original_image_2 = cv2.imread(image_path_2)

    # Itera sobre las coordenadas de los rectángulos y dibuja en la segunda imagen
    for (x, y, w, h) in rectangles_coordinates:
        # Dibuja el rectángulo en la segunda imagen
        cv2.rectangle(original_image_2, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imwrite('FinalImage.jpg', original_image_2)


    text_results = extract_text_from_rectangles()
    # Muestra los resultados
    for i, result in enumerate(text_results):
        print(f"Rectángulo {i + 1}:")
        print(f"Texto: {result['text']}")
        print(f"Vertices: {result['vertices']}")
        print()




if porcentaje_coincidencia >= 85:
    print(f"La imagen más similar es: {imagen_similar} con un {porcentaje_coincidencia:.2f}% de coincidencia.")
    second_step(ruta_referencia, imagen_similar)
        
else:
    print(f"No se encontró una coincidencia con suficiente porcentaje. Porcentaje de coincidencia: {porcentaje_coincidencia:.2f}% con la imagen {imagen_similar}")
