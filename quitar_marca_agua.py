import cv2
import numpy as np

# Cargar la imagen
ruta_imagen = 'C:\\Text_extraction\\Fotos_vertex\\images\\train\\Transaccion_PCH2.jpg'
imagen = cv2.imread(ruta_imagen)

# Aumentar el brillo (puedes ajustar el valor según tus necesidades)
factor_aumento_brillo = 1.5

# Convertir la imagen a un array NumPy de tipo float32
imagen_np = imagen.astype(np.float32)

# Multiplicar la imagen por el factor de aumento de brillo
imagen_aumentada_np = cv2.multiply(imagen_np, factor_aumento_brillo)

# Asegurarse de que los valores estén en el rango [0, 255]
imagen_aumentada_np = np.clip(imagen_aumentada_np, 0, 255).astype(np.uint8)

# Mostrar la imagen original y la aumentada
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Aumentada', imagen_aumentada_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
